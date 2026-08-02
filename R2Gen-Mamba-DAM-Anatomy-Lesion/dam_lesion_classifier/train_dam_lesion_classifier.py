#!/usr/bin/env python3

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)
from torchvision.models import (
    DenseNet121_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    densenet121,
    resnet18,
    resnet34,
    resnet50,
)
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

# Reuse the gated global-local DAM fusion module.
from models.r2gen import GatedGlobalLocalAdapter


PRIMARY_CLASSES = [
    "other_lesion",
    "lymph_node",
    "mass",
    "nodule",
    "cystic",
    "opacity",
    "consolidation",
]

ATTRIBUTE_CLASSES = [
    "none",
    "low_density",
    "enhancing",
    "calcified",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Reproducible behavior. This may be slightly slower.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_path(root: str, relative_or_absolute: str) -> str:
    path = Path(relative_or_absolute)

    if path.is_absolute():
        return str(path)

    return str(Path(root) / path)


def safe_torch_load(path: Path, device: torch.device) -> dict:
    """
    Loads a locally created checkpoint.

    weights_only=True is preferred when supported. The fallback keeps
    compatibility with older PyTorch releases.
    """
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def label_counts(
    rows: Sequence[dict],
    key: str,
    classes: Sequence[str],
) -> Dict[str, int]:
    counter = Counter()

    for row in rows:
        value = row.get(key)

        if value is None:
            raise KeyError(
                f"Missing label field {key!r} for ID "
                f"{row.get('id', '<unknown>')}"
            )

        if value not in classes:
            raise ValueError(
                f"Unexpected value {value!r} for field {key!r}; "
                f"expected one of {list(classes)}"
            )

        counter[value] += 1

    return {
        class_name: int(counter[class_name])
        for class_name in classes
    }


def make_class_weights(
    rows: Sequence[dict],
    key: str,
    classes: Sequence[str],
    power: float,
) -> torch.Tensor:
    """
    power=0.0: no class weighting.
    power=0.5: square-root inverse-frequency weighting.
    power=1.0: full inverse-frequency weighting.
    """
    counts = label_counts(rows, key, classes)

    count_tensor = torch.tensor(
        [max(counts[name], 1) for name in classes],
        dtype=torch.float32,
    )

    if power <= 0:
        return torch.ones_like(count_tensor)

    inverse = count_tensor.sum() / count_tensor
    weights = inverse.pow(power)

    # Normalize so the mean weight is 1.
    weights = weights / weights.mean()

    return weights


def make_sample_weights(
    rows: Sequence[dict],
    key: str,
    classes: Sequence[str],
    power: float,
) -> List[float]:
    """
    Optional weighted sampling.

    Keep power=0.0 when class-weighted CE is already used. Applying both
    aggressive weighted sampling and aggressive loss weighting can
    overpredict rare classes.
    """
    class_weights = make_class_weights(
        rows=rows,
        key=key,
        classes=classes,
        power=power,
    )

    mapping = {
        name: float(class_weights[i])
        for i, name in enumerate(classes)
    }

    return [
        mapping[row[key]]
        for row in rows
    ]


def replace_first_conv_with_four_channels(
    conv: nn.Conv2d,
) -> nn.Conv2d:
    """
    Converts a pretrained 3-channel first convolution into a 4-channel
    convolution. RGB weights are preserved. The mask channel is initialized
    from the mean RGB filter.
    """
    if conv.in_channels == 4:
        return conv

    if conv.in_channels != 3:
        raise ValueError(
            "Expected a pretrained 3-channel first convolution, "
            f"but got in_channels={conv.in_channels}"
        )

    new_conv = nn.Conv2d(
        in_channels=4,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )

    with torch.no_grad():
        new_conv.weight[:, :3].copy_(conv.weight)
        new_conv.weight[:, 3:4].copy_(
            conv.weight.mean(dim=1, keepdim=True)
        )

        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    return new_conv


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class DAMClassifierDataset(Dataset):
    def __init__(
        self,
        ann_path: str,
        image_dir: str,
        split: str,
        image_size: int = 224,
        crop_scale: float = 3.0,
        min_crop_size: int = 48,
        bbox_format: str = "xyxy",
        flip_prob: float = 0.5,
    ) -> None:
        super().__init__()

        with open(ann_path, "r") as handle:
            data = json.load(handle)

        if split not in data:
            raise KeyError(
                f"Split {split!r} not found in {ann_path}"
            )

        self.rows = data[split]
        self.image_dir = image_dir
        self.split = split
        self.image_size = image_size
        self.crop_scale = crop_scale
        self.min_crop_size = min_crop_size
        self.bbox_format = bbox_format
        self.flip_prob = flip_prob

        self.primary_to_index = {
            name: index
            for index, name in enumerate(PRIMARY_CLASSES)
        }

        self.attribute_to_index = {
            name: index
            for index, name in enumerate(ATTRIBUTE_CLASSES)
        }

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def first_box(row: dict) -> List[float]:
        box = row.get(
            "bboxes",
            row.get(
                "bbox",
                row.get("box"),
            ),
        )

        if isinstance(box, dict):
            box = box.get(
                "bbox",
                box.get("box"),
            )

        if (
            box
            and isinstance(box[0], (list, tuple))
        ):
            box = box[0]

        if box is None or len(box) != 4:
            raise ValueError(
                f"Invalid bbox for ID {row.get('id')}: {box}"
            )

        return [float(value) for value in box]

    def to_xyxy(
        self,
        box: Sequence[float],
        width: int,
        height: int,
    ) -> Tuple[float, float, float, float]:
        if self.bbox_format == "xyxy":
            x1, y1, x2, y2 = box

        elif self.bbox_format == "xywh":
            x, y, box_width, box_height = box
            x1 = x
            y1 = y
            x2 = x + box_width
            y2 = y + box_height

        elif self.bbox_format == "yolo":
            center_x, center_y, box_width, box_height = box

            x1 = (
                center_x - box_width / 2.0
            ) * width
            y1 = (
                center_y - box_height / 2.0
            ) * height
            x2 = (
                center_x + box_width / 2.0
            ) * width
            y2 = (
                center_y + box_height / 2.0
            ) * height

        else:
            raise ValueError(
                f"Unsupported bbox format: {self.bbox_format}"
            )

        x1 = max(0.0, min(width - 1.0, x1))
        y1 = max(0.0, min(height - 1.0, y1))
        x2 = max(x1 + 1.0, min(float(width), x2))
        y2 = max(y1 + 1.0, min(float(height), y2))

        return x1, y1, x2, y2

    def expand_box(
        self,
        box: Sequence[float],
        width: int,
        height: int,
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        original_width = x2 - x1
        original_height = y2 - y1

        crop_width = max(
            original_width,
            float(self.min_crop_size),
        ) * self.crop_scale

        crop_height = max(
            original_height,
            float(self.min_crop_size),
        ) * self.crop_scale

        crop_x1 = max(
            0,
            int(round(center_x - crop_width / 2.0)),
        )
        crop_y1 = max(
            0,
            int(round(center_y - crop_height / 2.0)),
        )
        crop_x2 = min(
            width,
            int(round(center_x + crop_width / 2.0)),
        )
        crop_y2 = min(
            height,
            int(round(center_y + crop_height / 2.0)),
        )

        crop_x2 = max(crop_x1 + 1, crop_x2)
        crop_y2 = max(crop_y1 + 1, crop_y2)

        return crop_x1, crop_y1, crop_x2, crop_y2

    def rgb_mask_tensor(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> torch.Tensor:
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )

        image_tensor = TF.to_tensor(image)

        image_tensor = TF.normalize(
            image_tensor,
            IMAGENET_MEAN,
            IMAGENET_STD,
        )

        mask_tensor = TF.to_tensor(mask)[:1]
        mask_tensor = (mask_tensor > 0.5).float()

        return torch.cat(
            [image_tensor, mask_tensor],
            dim=0,
        )

    def load_mask(
        self,
        row: dict,
        image_size: Tuple[int, int],
        box: Sequence[float],
    ) -> Image.Image:
        width, height = image_size
        mask_path = row.get("mask_path")

        if mask_path:
            full_mask_path = resolve_path(
                self.image_dir,
                mask_path,
            )

            if not os.path.exists(full_mask_path):
                raise FileNotFoundError(
                    f"Missing mask for ID {row.get('id')}: "
                    f"{full_mask_path}"
                )

            mask = Image.open(
                full_mask_path
            ).convert("L")

            if mask.size != image_size:
                raise ValueError(
                    f"Image/mask size mismatch for ID "
                    f"{row.get('id')}: image={image_size}, "
                    f"mask={mask.size}"
                )

            mask = mask.point(
                lambda value: 255 if value > 0 else 0,
                mode="L",
            )

            return mask

        # Fallback: create a rectangular mask from the bbox.
        x1, y1, x2, y2 = box

        mask = Image.new(
            "L",
            (width, height),
            0,
        )

        draw = ImageDraw.Draw(mask)

        draw.rectangle(
            [
                round(x1),
                round(y1),
                max(round(x1), round(x2) - 1),
                max(round(y1), round(y2) - 1),
            ],
            fill=255,
        )

        return mask

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[str, torch.Tensor, int, int]:
        row = self.rows[index]

        image_path = row["image_path"]

        if isinstance(image_path, (list, tuple)):
            image_path = image_path[0]

        full_image_path = resolve_path(
            self.image_dir,
            image_path,
        )

        if not os.path.exists(full_image_path):
            raise FileNotFoundError(
                f"Missing image for ID {row.get('id')}: "
                f"{full_image_path}"
            )

        image = Image.open(
            full_image_path
        ).convert("RGB")

        width, height = image.size

        box = self.to_xyxy(
            self.first_box(row),
            width,
            height,
        )

        mask = self.load_mask(
            row=row,
            image_size=image.size,
            box=box,
        )

        crop_box = self.expand_box(
            box,
            width,
            height,
        )

        focal_image = image.crop(crop_box)
        focal_mask = mask.crop(crop_box)

        if (
            self.split == "train"
            and random.random() < self.flip_prob
        ):
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            focal_image = TF.hflip(focal_image)
            focal_mask = TF.hflip(focal_mask)

        global_input = self.rgb_mask_tensor(
            image,
            mask,
        )

        focal_input = self.rgb_mask_tensor(
            focal_image,
            focal_mask,
        )

        images = torch.stack(
            [global_input, focal_input],
            dim=0,
        )

        # Prefer the explicit classifier fields created by the corrected
        # preparation script.
        primary_name = row.get(
            "classifier_primary_target",
            row.get("oracle_primary_lesion_type"),
        )

        attribute_name = row.get(
            "classifier_attribute_target",
            row.get("oracle_lesion_attribute"),
        )

        if primary_name not in self.primary_to_index:
            raise ValueError(
                f"Invalid primary target {primary_name!r} "
                f"for ID {row.get('id')}"
            )

        if attribute_name not in self.attribute_to_index:
            raise ValueError(
                f"Invalid attribute target {attribute_name!r} "
                f"for ID {row.get('id')}"
            )

        primary_index = self.primary_to_index[
            primary_name
        ]

        attribute_index = self.attribute_to_index[
            attribute_name
        ]

        return (
            str(row["id"]),
            images,
            primary_index,
            attribute_index,
        )


# ---------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------

class FeatureBackbone(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: bool,
    ) -> None:
        super().__init__()

        self.name = name.lower()

        if self.name == "resnet18":
            weights = (
                ResNet18_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )

            model = resnet18(weights=weights)

            model.conv1 = replace_first_conv_with_four_channels(
                model.conv1
            )

            self.features = nn.Sequential(
                *list(model.children())[:-2]
            )

            self.output_dim = 512

        elif self.name == "resnet34":
            weights = (
                ResNet34_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )

            model = resnet34(weights=weights)

            model.conv1 = replace_first_conv_with_four_channels(
                model.conv1
            )

            self.features = nn.Sequential(
                *list(model.children())[:-2]
            )

            self.output_dim = 512

        elif self.name == "resnet50":
            weights = (
                ResNet50_Weights.IMAGENET1K_V2
                if pretrained
                else None
            )

            model = resnet50(weights=weights)

            model.conv1 = replace_first_conv_with_four_channels(
                model.conv1
            )

            self.features = nn.Sequential(
                *list(model.children())[:-2]
            )

            self.output_dim = 2048

        elif self.name == "densenet121":
            weights = (
                DenseNet121_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )

            model = densenet121(weights=weights)

            model.features.conv0 = (
                replace_first_conv_with_four_channels(
                    model.features.conv0
                )
            )

            self.features = model.features
            self.output_dim = 1024

        else:
            raise ValueError(
                f"Unsupported backbone {name!r}. "
                "Use resnet18, resnet34, resnet50, "
                "or densenet121."
            )

    def forward(
        self,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.features(image)

        if self.name == "densenet121":
            feature_map = F.relu(
                feature_map,
                inplace=False,
            )

        tokens = feature_map.flatten(2).transpose(1, 2)

        pooled = F.adaptive_avg_pool2d(
            feature_map,
            output_size=1,
        ).flatten(1)

        return tokens, pooled


# ---------------------------------------------------------------------
# DAM classifier model
# ---------------------------------------------------------------------

class DAMLesionClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        adapter_dim: int = 256,
        heads: int = 8,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()

        self.backbone_name = backbone

        # Shared weights for global and focal streams.
        self.encoder = FeatureBackbone(
            name=backbone,
            pretrained=pretrained,
        )

        feature_dim = self.encoder.output_dim

        if feature_dim % heads != 0:
            raise ValueError(
                f"feature_dim={feature_dim} must be divisible "
                f"by num_heads={heads}"
            )

        self.adapter = GatedGlobalLocalAdapter(
            feature_dim,
            adapter_dim,
            heads,
            dropout,
        )

        # Combine attention-fused local features with pooled global
        # and local features.
        fusion_dim = feature_dim * 3

        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.primary_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(
                feature_dim,
                len(PRIMARY_CLASSES),
            ),
        )

        self.attribute_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(
                feature_dim,
                len(ATTRIBUTE_CLASSES),
            ),
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError(
                "Expected images with shape [B, 2, 4, H, W], "
                f"got {tuple(images.shape)}"
            )

        if images.shape[1] != 2:
            raise ValueError(
                "Expected two views: global and focal; "
                f"got shape {tuple(images.shape)}"
            )

        if images.shape[2] != 4:
            raise ValueError(
                "Expected four channels: RGB + mask; "
                f"got shape {tuple(images.shape)}"
            )

        global_tokens, global_pooled = self.encoder(
            images[:, 0]
        )

        local_tokens, local_pooled = self.encoder(
            images[:, 1]
        )

        fused_local_tokens = self.adapter(
            local_tokens,
            global_tokens,
        )

        attention_pooled = fused_local_tokens.mean(
            dim=1
        )

        combined = torch.cat(
            [
                attention_pooled,
                global_pooled,
                local_pooled,
            ],
            dim=1,
        )

        fused_feature = self.fusion(combined)

        primary_logits = self.primary_head(
            fused_feature
        )

        attribute_logits = self.attribute_head(
            fused_feature
        )

        return primary_logits, attribute_logits


# ---------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    primary_criterion: nn.Module,
    attribute_criterion: nn.Module,
    attribute_lambda: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: float = 0.0,
) -> Tuple[dict, dict]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0

    all_ids: List[str] = []

    true_primary: List[int] = []
    predicted_primary: List[int] = []
    primary_probabilities: List[List[float]] = []

    true_attribute: List[int] = []
    predicted_attribute: List[int] = []
    attribute_probabilities: List[List[float]] = []

    for (
        sample_ids,
        images,
        primary_targets,
        attribute_targets,
    ) in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        primary_targets = primary_targets.to(
            device,
            non_blocking=True,
        )

        attribute_targets = attribute_targets.to(
            device,
            non_blocking=True,
        )

        with torch.set_grad_enabled(is_training):
            (
                primary_logits,
                attribute_logits,
            ) = model(images)

            primary_loss = primary_criterion(
                primary_logits,
                primary_targets,
            )

            attribute_loss = attribute_criterion(
                attribute_logits,
                attribute_targets,
            )

            loss = (
                primary_loss
                + attribute_lambda * attribute_loss
            )

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=grad_clip,
                    )

                optimizer.step()

        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size

        primary_probs = torch.softmax(
            primary_logits,
            dim=1,
        )

        attribute_probs = torch.softmax(
            attribute_logits,
            dim=1,
        )

        primary_predictions = primary_probs.argmax(
            dim=1
        )

        attribute_predictions = attribute_probs.argmax(
            dim=1
        )

        all_ids.extend(list(sample_ids))

        true_primary.extend(
            primary_targets.detach().cpu().tolist()
        )

        predicted_primary.extend(
            primary_predictions.detach().cpu().tolist()
        )

        primary_probabilities.extend(
            primary_probs.detach().cpu().tolist()
        )

        true_attribute.extend(
            attribute_targets.detach().cpu().tolist()
        )

        predicted_attribute.extend(
            attribute_predictions.detach().cpu().tolist()
        )

        attribute_probabilities.extend(
            attribute_probs.detach().cpu().tolist()
        )

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "primary_acc": accuracy_score(
            true_primary,
            predicted_primary,
        ),
        "primary_macro_f1": f1_score(
            true_primary,
            predicted_primary,
            average="macro",
            zero_division=0,
        ),
        "attribute_acc": accuracy_score(
            true_attribute,
            predicted_attribute,
        ),
        "attribute_macro_f1": f1_score(
            true_attribute,
            predicted_attribute,
            average="macro",
            zero_division=0,
        ),
    }

    outputs = {
        "ids": all_ids,
        "true_primary": true_primary,
        "predicted_primary": predicted_primary,
        "primary_probabilities": primary_probabilities,
        "true_attribute": true_attribute,
        "predicted_attribute": predicted_attribute,
        "attribute_probabilities": attribute_probabilities,
    }

    return metrics, outputs


def save_predictions(
    output_path: Path,
    outputs: dict,
) -> None:
    with open(
        output_path,
        "w",
        newline="",
    ) as handle:
        writer = csv.writer(handle)

        header = [
            "id",
            "true_primary",
            "predicted_primary",
            "primary_confidence",
            "true_attribute",
            "predicted_attribute",
            "attribute_confidence",
        ]

        for name in PRIMARY_CLASSES:
            header.append(
                f"primary_probability_{name}"
            )

        for name in ATTRIBUTE_CLASSES:
            header.append(
                f"attribute_probability_{name}"
            )

        writer.writerow(header)

        for index, sample_id in enumerate(
            outputs["ids"]
        ):
            true_primary_index = outputs[
                "true_primary"
            ][index]

            predicted_primary_index = outputs[
                "predicted_primary"
            ][index]

            primary_probs = outputs[
                "primary_probabilities"
            ][index]

            true_attribute_index = outputs[
                "true_attribute"
            ][index]

            predicted_attribute_index = outputs[
                "predicted_attribute"
            ][index]

            attribute_probs = outputs[
                "attribute_probabilities"
            ][index]

            row = [
                sample_id,
                PRIMARY_CLASSES[true_primary_index],
                PRIMARY_CLASSES[
                    predicted_primary_index
                ],
                max(primary_probs),
                ATTRIBUTE_CLASSES[
                    true_attribute_index
                ],
                ATTRIBUTE_CLASSES[
                    predicted_attribute_index
                ],
                max(attribute_probs),
            ]

            row.extend(primary_probs)
            row.extend(attribute_probs)

            writer.writerow(row)


def save_confusion_matrix(
    output_path: Path,
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    classes: Sequence[str],
) -> None:
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(len(classes))),
    )

    with open(
        output_path,
        "w",
        newline="",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow(
            ["true\\predicted"] + list(classes)
        )

        for class_name, row in zip(
            classes,
            matrix,
        ):
            writer.writerow(
                [class_name] + row.tolist()
            )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_score: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "args": vars(args),
            "classes": {
                "primary": PRIMARY_CLASSES,
                "attribute": ATTRIBUTE_CLASSES,
            },
        },
        path,
    )


def build_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
    sampler=None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DAM global/focal RGB+mask dual-head "
            "DeepLesion classifier"
        )
    )

    parser.add_argument(
        "--ann_path",
        required=True,
    )

    parser.add_argument(
        "--image_dir",
        required=True,
    )

    parser.add_argument(
        "--save_dir",
        required=True,
    )

    parser.add_argument(
        "--backbone",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "densenet121",
        ],
        default="resnet18",
    )

    parser.add_argument(
        "--no_pretrained",
        action="store_true",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--attribute_lambda",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--primary_weight_power",
        type=float,
        default=0.5,
        help=(
            "0=no weighting, 0.5=sqrt inverse frequency, "
            "1=full inverse frequency"
        ),
    )

    parser.add_argument(
        "--attribute_weight_power",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--sampler_power",
        type=float,
        default=0.0,
        help=(
            "0 disables weighted sampling. Avoid using a large "
            "sampler power together with large CE class weights."
        ),
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--crop_scale",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--min_crop_size",
        type=int,
        default=48,
    )

    parser.add_argument(
        "--bbox_format",
        choices=[
            "xyxy",
            "xywh",
            "yolo",
        ],
        default="xyxy",
    )

    parser.add_argument(
        "--flip_prob",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--adapter_dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--lr_patience",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--lr_factor",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-7,
    )

    parser.add_argument(
        "--grad_clip",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=9223,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_seed(args.seed)

    with open(args.ann_path, "r") as handle:
        raw_data = json.load(handle)

    train_rows = raw_data["train"]

    print("Configuration:")
    print(json.dumps(vars(args), indent=2))

    print("\nDataset sizes:")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {len(raw_data[split])}")

    print("\nPrimary training distribution:")
    primary_counts = label_counts(
        train_rows,
        "classifier_primary_target",
        PRIMARY_CLASSES,
    )
    print(json.dumps(primary_counts, indent=2))

    print("\nAttribute training distribution:")
    attribute_counts = label_counts(
        train_rows,
        "classifier_attribute_target",
        ATTRIBUTE_CLASSES,
    )
    print(json.dumps(attribute_counts, indent=2))

    train_dataset = DAMClassifierDataset(
        ann_path=args.ann_path,
        image_dir=args.image_dir,
        split="train",
        image_size=args.image_size,
        crop_scale=args.crop_scale,
        min_crop_size=args.min_crop_size,
        bbox_format=args.bbox_format,
        flip_prob=args.flip_prob,
    )

    val_dataset = DAMClassifierDataset(
        ann_path=args.ann_path,
        image_dir=args.image_dir,
        split="val",
        image_size=args.image_size,
        crop_scale=args.crop_scale,
        min_crop_size=args.min_crop_size,
        bbox_format=args.bbox_format,
        flip_prob=0.0,
    )

    test_dataset = DAMClassifierDataset(
        ann_path=args.ann_path,
        image_dir=args.image_dir,
        split="test",
        image_size=args.image_size,
        crop_scale=args.crop_scale,
        min_crop_size=args.min_crop_size,
        bbox_format=args.bbox_format,
        flip_prob=0.0,
    )

    sampler = None

    if args.sampler_power > 0:
        sample_weights = make_sample_weights(
            rows=train_rows,
            key="classifier_primary_target",
            classes=PRIMARY_CLASSES,
            power=args.sampler_power,
        )

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        print(
            "\nWeighted sampler enabled with power:",
            args.sampler_power,
        )
    else:
        print("\nWeighted sampler disabled.")

    train_loader = build_loader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=sampler is None,
        sampler=sampler,
    )

    val_loader = build_loader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    test_loader = build_loader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:", device)

    model = DAMLesionClassifier(
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        adapter_dim=args.adapter_dim,
        heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    primary_weights = make_class_weights(
        rows=train_rows,
        key="classifier_primary_target",
        classes=PRIMARY_CLASSES,
        power=args.primary_weight_power,
    ).to(device)

    attribute_weights = make_class_weights(
        rows=train_rows,
        key="classifier_attribute_target",
        classes=ATTRIBUTE_CLASSES,
        power=args.attribute_weight_power,
    ).to(device)

    print("\nPrimary CE weights:")
    for name, value in zip(
        PRIMARY_CLASSES,
        primary_weights.detach().cpu().tolist(),
    ):
        print(f"  {name:20s}: {value:.6f}")

    print("\nAttribute CE weights:")
    for name, value in zip(
        ATTRIBUTE_CLASSES,
        attribute_weights.detach().cpu().tolist(),
    ):
        print(f"  {name:20s}: {value:.6f}")

    primary_criterion = nn.CrossEntropyLoss(
        weight=primary_weights
    )

    attribute_criterion = nn.CrossEntropyLoss(
        weight=attribute_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    best_path = save_dir / "best.pt"
    last_path = save_dir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics, _ = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            primary_criterion=primary_criterion,
            attribute_criterion=attribute_criterion,
            attribute_lambda=args.attribute_lambda,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )

        val_metrics, _ = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            primary_criterion=primary_criterion,
            attribute_criterion=attribute_criterion,
            attribute_lambda=args.attribute_lambda,
            optimizer=None,
            grad_clip=0.0,
        )

        validation_score = val_metrics[
            "primary_macro_f1"
        ]

        scheduler.step(validation_score)

        current_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "lr": current_lr,
            "train": train_metrics,
            "val": val_metrics,
        }

        history.append(record)

        print(
            f"epoch {epoch} "
            f"lr {current_lr:.8g} "
            f"train {train_metrics} "
            f"val {val_metrics}",
            flush=True,
        )

        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_score=best_score,
            args=args,
        )

        if validation_score > best_score:
            best_score = validation_score
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_score=best_score,
                args=args,
            )

            print(
                f"Saved new best checkpoint: epoch={epoch}, "
                f"val_primary_macro_f1={best_score:.6f}",
                flush=True,
            )

        else:
            epochs_without_improvement += 1

            print(
                "No improvement:",
                epochs_without_improvement,
                "/",
                args.patience,
                flush=True,
            )

            if (
                epochs_without_improvement
                >= args.patience
            ):
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best epoch was {best_epoch}.",
                    flush=True,
                )
                break

        with open(
            save_dir / "history.json",
            "w",
        ) as handle:
            json.dump(
                history,
                handle,
                indent=2,
            )

    checkpoint = safe_torch_load(
        best_path,
        device,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    print(
        "\nLoaded best checkpoint:",
        best_path,
    )
    print(
        "Best epoch:",
        checkpoint.get("epoch"),
    )
    print(
        "Best validation primary macro F1:",
        checkpoint.get("best_score"),
    )

    test_metrics, test_outputs = run_epoch(
        model=model,
        loader=test_loader,
        device=device,
        primary_criterion=primary_criterion,
        attribute_criterion=attribute_criterion,
        attribute_lambda=args.attribute_lambda,
        optimizer=None,
        grad_clip=0.0,
    )

    print("\nTEST", test_metrics)

    primary_report_text = classification_report(
        test_outputs["true_primary"],
        test_outputs["predicted_primary"],
        labels=list(range(len(PRIMARY_CLASSES))),
        target_names=PRIMARY_CLASSES,
        digits=4,
        zero_division=0,
    )

    attribute_report_text = classification_report(
        test_outputs["true_attribute"],
        test_outputs["predicted_attribute"],
        labels=list(range(len(ATTRIBUTE_CLASSES))),
        target_names=ATTRIBUTE_CLASSES,
        digits=4,
        zero_division=0,
    )

    primary_report_dict = classification_report(
        test_outputs["true_primary"],
        test_outputs["predicted_primary"],
        labels=list(range(len(PRIMARY_CLASSES))),
        target_names=PRIMARY_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    attribute_report_dict = classification_report(
        test_outputs["true_attribute"],
        test_outputs["predicted_attribute"],
        labels=list(range(len(ATTRIBUTE_CLASSES))),
        target_names=ATTRIBUTE_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    print("\nPRIMARY\n")
    print(primary_report_text)

    print("\nATTRIBUTE\n")
    print(attribute_report_text)

    final_results = {
        "best_epoch": checkpoint.get("epoch"),
        "best_val_primary_macro_f1": checkpoint.get(
            "best_score"
        ),
        "test_metrics": test_metrics,
        "primary_classification_report": (
            primary_report_dict
        ),
        "attribute_classification_report": (
            attribute_report_dict
        ),
        "configuration": vars(args),
    }

    with open(
        save_dir / "test_metrics.json",
        "w",
    ) as handle:
        json.dump(
            final_results,
            handle,
            indent=2,
        )

    with open(
        save_dir / "primary_report.txt",
        "w",
    ) as handle:
        handle.write(primary_report_text)

    with open(
        save_dir / "attribute_report.txt",
        "w",
    ) as handle:
        handle.write(attribute_report_text)

    save_predictions(
        output_path=save_dir / "test_predictions.csv",
        outputs=test_outputs,
    )

    save_confusion_matrix(
        output_path=(
            save_dir
            / "primary_confusion_matrix.csv"
        ),
        true_labels=test_outputs["true_primary"],
        predicted_labels=test_outputs[
            "predicted_primary"
        ],
        classes=PRIMARY_CLASSES,
    )

    save_confusion_matrix(
        output_path=(
            save_dir
            / "attribute_confusion_matrix.csv"
        ),
        true_labels=test_outputs["true_attribute"],
        predicted_labels=test_outputs[
            "predicted_attribute"
        ],
        classes=ATTRIBUTE_CLASSES,
    )

    print("\nSaved outputs to:", save_dir)
    print("  best.pt")
    print("  last.pt")
    print("  history.json")
    print("  test_metrics.json")
    print("  test_predictions.csv")
    print("  primary_report.txt")
    print("  attribute_report.txt")
    print("  primary_confusion_matrix.csv")
    print("  attribute_confusion_matrix.csv")


if __name__ == "__main__":
    main()
