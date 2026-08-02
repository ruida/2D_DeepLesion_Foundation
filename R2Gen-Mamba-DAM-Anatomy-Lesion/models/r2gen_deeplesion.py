import contextlib
import torch
import torch.nn as nn

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder
from modules.lesion_encoder import LesionFusion


@contextlib.contextmanager
def _bypass_att_embed(enc_dec):
    original = enc_dec.att_embed
    enc_dec.att_embed = nn.Identity()
    try:
        yield
    finally:
        enc_dec.att_embed = original


class GatedGlobalLocalAdapter(nn.Module):
    """DAM-inspired global-to-focal cross-attention with zero gates."""
    def __init__(self, input_dim=2048, adapter_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        if adapter_dim % num_heads != 0:
            raise ValueError("dam_adapter_dim must be divisible by dam_num_heads")
        self.local_norm = nn.LayerNorm(input_dim)
        self.global_norm = nn.LayerNorm(input_dim)
        self.local_down = nn.Linear(input_dim, adapter_dim)
        self.global_down = nn.Linear(input_dim, adapter_dim)
        self.cross_attn = nn.MultiheadAttention(adapter_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(adapter_dim)
        self.ffn = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(adapter_dim * 2, adapter_dim),
        )
        self.up = nn.Linear(adapter_dim, input_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, local_tokens, global_tokens):
        base = self.local_down(self.local_norm(local_tokens))
        glob = self.global_down(self.global_norm(global_tokens))
        cross, _ = self.cross_attn(base, glob, glob, need_weights=False)
        fused = base + torch.tanh(self.gamma) * cross
        fused = fused + torch.tanh(self.beta) * self.ffn(self.ffn_norm(fused))
        return local_tokens + self.up(fused - base)


class R2GenDeepLesionModel(nn.Module):
    """Latest merged model: DAM visual prompt + bbox/anatomy/lesion tokens."""
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.use_dam = getattr(args, "use_dam", False)
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        self.lesion_fusion = LesionFusion(
            d_model=args.d_model,
            num_anatomy=getattr(args, "num_anatomy", 174),
            max_boxes=getattr(args, "max_boxes", 2),
            max_anatomy=getattr(args, "max_anatomy", 20),
            bbox_hidden=getattr(args, "bbox_hidden", 128),
            anatomy_encoding=getattr(args, "anatomy_encoding", "id"),
            anatomy_text_vocab_size=tokenizer.get_vocab_size() + 1,
        )
        if self.use_dam:
            self.dam_adapter = GatedGlobalLocalAdapter(
                input_dim=args.d_vf,
                adapter_dim=getattr(args, "dam_adapter_dim", 512),
                num_heads=getattr(args, "dam_num_heads", 8),
                dropout=args.dropout,
            )
            self.dam_output_mode = getattr(args, "dam_output_mode", "local")

    def __str__(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return super().__str__() + f"\nTrainable parameters: {n}"

    def _extract_visual(self, images):
        if self.use_dam:
            if images.ndim != 5 or images.size(1) != 2 or images.size(2) != 4:
                raise ValueError("DAM DeepLesion expects [B,2,4,H,W]")
            global_tokens, global_fc = self.visual_extractor(images[:, 0])
            local_tokens, local_fc = self.visual_extractor(images[:, 1])
            fused_local = self.dam_adapter(local_tokens, global_tokens)
            if self.dam_output_mode == "concat":
                return torch.cat([global_tokens, fused_local], 1), 0.5 * (global_fc + local_fc)
            if self.dam_output_mode == "local":
                return fused_local, local_fc
            raise ValueError("dam_output_mode must be local or concat")

        att0, fc0 = self.visual_extractor(images[:, 0])
        att1, fc1 = self.visual_extractor(images[:, 1])
        return torch.cat([att0, att1], 1), torch.cat([fc0, fc1], 1)

    def _project_visual(self, att_feats):
        b, s, _ = att_feats.shape
        return self.encoder_decoder.att_embed(att_feats.reshape(b*s, -1)).reshape(b, s, -1)

    def forward(self, images, targets=None, mode="train", bboxes=None, bbox_mask=None, anatomy_ids=None):
        att_feats, fc_feats = self._extract_visual(images)
        use_conditioning = bboxes is not None and bbox_mask is not None and anatomy_ids is not None
        if use_conditioning:
            projected = self._project_visual(att_feats)
            fused_feats, _ = self.lesion_fusion(projected, bboxes, bbox_mask, anatomy_ids)
            with _bypass_att_embed(self.encoder_decoder):
                if mode == "train":
                    return self.encoder_decoder(fc_feats, fused_feats, targets, mode="forward", att_masks=None)
                output, _ = self.encoder_decoder(fc_feats, fused_feats, mode="sample", att_masks=None)
                return output
        if mode == "train":
            return self.encoder_decoder(fc_feats, att_feats, targets, mode="forward")
        output, _ = self.encoder_decoder(fc_feats, att_feats, mode="sample")
        return output
