import torch
import torch.nn as nn
import torchvision.models as models


class VisualExtractor(nn.Module):
    """Shared ResNet extractor with optional aligned RGB+mask input.

    With --use_dam, conv1 accepts four channels. The RGB weights are copied
    from ImageNet and the mask channel is initialized to zero, preserving the
    pretrained RGB behavior at initialization. The same backbone is used for
    full-image and focal-crop inputs.
    """

    def __init__(self, args):
        super().__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        self.use_dam = getattr(args, "use_dam", False)

        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        if self.use_dam:
            old_conv = model.conv1
            new_conv = nn.Conv2d(
                4, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                stride=old_conv.stride, padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )
            with torch.no_grad():
                new_conv.weight[:, :3].copy_(old_conv.weight)
                new_conv.weight[:, 3:].zero_()
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
            model.conv1 = new_conv

        self.model = nn.Sequential(*list(model.children())[:-2])
        self.avg_fnt = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images):
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).flatten(1)
        b, c, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(b, c, -1).permute(0, 2, 1)
        return patch_feats, avg_feats
