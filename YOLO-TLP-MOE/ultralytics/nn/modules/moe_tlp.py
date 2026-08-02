import copy
import weakref
import torch
import torch.nn as nn
import torch.nn.functional as F

# Auxiliary-loss registry to avoid deepcopy issues with non-leaf tensors
MOE_LOSS_REGISTRY = weakref.WeakKeyDictionary()


def _robust_deepcopy(obj, memo):
    cls = obj.__class__
    new_obj = cls.__new__(cls)
    memo[id(obj)] = new_obj
    for k, v in obj.__dict__.items():
        if isinstance(v, torch.Tensor) and v.grad_fn is not None:
            setattr(new_obj, k, torch.tensor(0.0))
        else:
            try:
                setattr(new_obj, k, copy.deepcopy(v, memo))
            except Exception:
                setattr(new_obj, k, v)
    return new_obj


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class EfficientExpertGroup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        return self.conv(x)


class DynamicRoutingLayer(nn.Module):
    """
    Global router used by ES_MOE.
    Training: masked soft top-k to preserve gradients.
    Inference: hard top-k for sparse routing.
    """
    def __init__(self, in_channels, num_experts=3, reduction=8, top_k=None):
        super().__init__()
        reduced_channels = max(in_channels // reduction, 8)
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts) if top_k is not None else num_experts
        self.use_top_k = top_k is not None
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.routing_network = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, num_experts, kernel_size=1),
        )

    def forward(self, x):
        pooled = self.global_pool(x)
        routing_logits = self.routing_network(pooled)
        if not self.use_top_k:
            routing_weights = F.softmax(routing_logits.float(), dim=1).type_as(x)
        elif self.training:
            routing_weights = self._soft_top_k(routing_logits)
        else:
            routing_weights = self._hard_top_k(routing_logits)
        return routing_weights.repeat(1, 1, x.size(2), x.size(3))

    def _soft_top_k(self, logits):
        B, E, H, W = logits.shape
        logits_flat = logits.view(B, E, -1).clamp(-30.0, 30.0)
        weights = F.softmax(logits_flat.float(), dim=1).type_as(logits)
        _, topk_indices = torch.topk(weights, self.top_k, dim=1)
        idx = topk_indices.permute(0, 2, 1).contiguous()
        mask_one_hot = F.one_hot(idx, num_classes=E).sum(dim=2)
        mask_one_hot = mask_one_hot.permute(0, 2, 1).contiguous().to(weights.dtype)
        weights = weights * mask_one_hot
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
        return weights.view(B, E, H, W)

    def _hard_top_k(self, logits):
        B, E, H, W = logits.shape
        logits_flat = logits.view(B, E, -1)
        topk_values, topk_indices = torch.topk(logits_flat, self.top_k, dim=1)
        topk_values = topk_values.clamp(-30.0, 30.0)
        topk_weights = F.softmax(topk_values.float(), dim=1).type_as(logits)
        idx = topk_indices.permute(0, 2, 1).contiguous()
        oh = F.one_hot(idx, num_classes=E)
        tw = topk_weights.permute(0, 2, 1).contiguous()
        weighted = (oh.to(tw.dtype) * tw.unsqueeze(-1)).sum(dim=2)
        weights = weighted.permute(0, 2, 1).contiguous()
        return weights.view(B, E, H, W)


class ES_MOE(nn.Module):
    """
    Minimal ES-MoE block ported for YOLO-TLP backbone insertion.
    Uses multi-kernel depthwise-separable experts + global dynamic router.
    """
    def __init__(
        self,
        in_channels,
        out_channels=None,
        num_experts=4,
        reduction=8,
        top_k=2,
        use_sparse_inference=True,
        dynamic_threshold=0.0,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts) if top_k is not None else num_experts
        self.use_top_k = top_k is not None
        self.use_sparse_inference = use_sparse_inference
        self.dynamic_threshold = dynamic_threshold

        self.routing = DynamicRoutingLayer(in_channels, num_experts, reduction, top_k)

        kernel_sizes = [3, 5, 7, 9]
        if num_experts <= len(kernel_sizes):
            ks = kernel_sizes[:num_experts]
        else:
            ks = [3 + 2 * i for i in range(num_experts)]
        self.experts = nn.ModuleList([
            EfficientExpertGroup(in_channels, out_channels, kernel_size=k) for k in ks
        ])
        self.norm = nn.Sequential(nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True))

    def forward(self, x):
        routing_weights = self.routing(x)
        self._compute_load_balancing_loss(routing_weights)

        if self.training or not self.use_top_k or not self.use_sparse_inference:
            out = self._dense_forward(x, routing_weights)
        else:
            out = self._sparse_forward(x, routing_weights.detach())

        out = self.norm(out)
        if self.in_channels == self.out_channels:
            out = out + x
        return out

    def _dense_forward(self, x, routing_weights):
        out = 0
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            weight = routing_weights[:, i:i + 1, :, :]
            out = out + expert_out * weight
        return out

    def _sparse_forward(self, x, routing_weights):
        B, E, H, W = routing_weights.shape
        expert_importance = routing_weights.view(B, E, -1).mean(dim=2)
        _, topk_indices = torch.topk(expert_importance, self.top_k, dim=1)
        final_output = torch.zeros(B, self.out_channels, H, W, device=x.device, dtype=x.dtype)

        for expert_idx in range(self.num_experts):
            mask = topk_indices == expert_idx
            if not mask.any():
                continue
            batch_indices, k_ranks = torch.where(mask)
            if self.dynamic_threshold > 0:
                current_weights = routing_weights[batch_indices, expert_idx:expert_idx + 1, :, :]
                weight_means = current_weights.mean(dim=(1, 2, 3))
                keep_mask = (k_ranks == 0) | (weight_means >= self.dynamic_threshold)
                batch_indices = batch_indices[keep_mask]
                if batch_indices.numel() == 0:
                    continue
            expert_out = self.experts[expert_idx](x[batch_indices])
            weight = routing_weights[batch_indices, expert_idx:expert_idx + 1, :, :]
            final_output.index_add_(0, batch_indices, expert_out * weight)
        return final_output

    def _compute_load_balancing_loss(self, routing_weights, eps=1e-6):
        expert_usage = routing_weights.mean(dim=(0, 2, 3))
        ideal_usage = 1.0 / self.num_experts
        load_balance_loss = F.mse_loss(expert_usage, torch.full_like(expert_usage, ideal_usage))
        if torch.isnan(load_balance_loss):
            load_balance_loss = torch.tensor(0.0, device=routing_weights.device, requires_grad=True)
        MOE_LOSS_REGISTRY[self] = load_balance_loss
        return load_balance_loss

    @property
    def aux_loss(self):
        return MOE_LOSS_REGISTRY.get(self, torch.tensor(0.0, device=next(self.parameters()).device))

    def __deepcopy__(self, memo):
        return _robust_deepcopy(self, memo)
