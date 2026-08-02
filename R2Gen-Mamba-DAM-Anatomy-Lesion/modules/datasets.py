import os
import json
import random
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ROUGH_ANATOMY_ID_TO_NAME = {
    1: "lung", 2: "liver", 3: "kidney", 4: "adrenal",
    5: "lymph node", 6: "bone", 7: "soft tissue", 8: "abdomen",
    9: "pelvis", 10: "chest", 11: "brain head neck", 12: "spine",
    13: "lesion",
}


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        with open(self.ann_path, "r") as f:
            self.ann = json.load(f)
        self.examples = self.ann[self.split]
        for ex in self.examples:
            ex["ids"] = tokenizer(ex["report"])[:self.max_seq_length]
            ex["mask"] = [1] * len(ex["ids"])

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        ex = self.examples[idx]
        p = ex["image_path"]
        a = Image.open(os.path.join(self.image_dir, p[0])).convert("RGB")
        b = Image.open(os.path.join(self.image_dir, p[1])).convert("RGB")
        if self.transform is not None:
            a, b = self.transform(a), self.transform(b)
        return ex["id"], torch.stack((a, b), 0), ex["ids"], ex["mask"], len(ex["ids"])


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        ex = self.examples[idx]
        image = Image.open(os.path.join(self.image_dir, ex["image_path"][0])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return ex["id"], image, ex["ids"], ex["mask"], len(ex["ids"])


class DeepLesionDataset(BaseDataset):
    """DeepLesion dataset supporting both legacy dual-RGB and merged DAM inputs.

    When --use_dam is enabled, the returned image tensor is [2,4,H,W]:
      0 = full RGB image + aligned full mask
      1 = focal RGB crop + aligned focal mask

    The same sample also returns normalized bbox tokens plus anatomy and lesion-
    type IDs for R2Gen-Mamba conditioning.

    Supported metadata fields include:
      anatomy_ids, rough_anatomy_id(s), rough_anatomy_name(s),
      lesion_type_id, lesion_type_ids, mask_path, bboxes.
    """

    MAX_BOXES = 2
    MAX_ANATOMY = 20

    def __init__(self, args, tokenizer, split, transform=None):
        super().__init__(args, tokenizer, split, transform=None if getattr(args, "use_dam", False) else transform)
        self.max_boxes = getattr(args, "max_boxes", self.MAX_BOXES)
        self.max_anatomy = getattr(args, "max_anatomy", self.MAX_ANATOMY)
        self.anatomy_source = getattr(args, "anatomy_source", "old")
        self.anatomy_encoding = getattr(args, "anatomy_encoding", "id")
        self.include_lesion_type = getattr(args, "include_lesion_type", False)
        self.use_dam = getattr(args, "use_dam", False)
        self.image_size = getattr(args, "dam_image_size", 224)
        self.crop_scale = getattr(args, "dam_crop_scale", 3.0)
        self.min_crop_size = getattr(args, "dam_min_crop_size", 48)
        self.bbox_format = getattr(args, "bbox_format", "xyxy")
        self.train_flip_prob = getattr(args, "dam_flip_prob", 0.5)

    def _resolve(self, path):
        if os.path.isabs(path):
            return path
        return os.path.join(self.image_dir, path)

    def _load_rgb(self, path):
        full = self._resolve(path)
        if not os.path.isfile(full):
            raise FileNotFoundError(f"Image not found: {full}")
        return Image.open(full).convert("RGB")

    @staticmethod
    def _first_box(ex):
        boxes = ex.get("dam_bboxes", ex.get("bboxes", ex.get("bbox", ex.get("box"))))
        if boxes is None:
            raise KeyError(f"Sample {ex.get('id')} has no bbox field")
        if isinstance(boxes, dict):
            boxes = boxes.get("bbox", boxes.get("box"))
        if len(boxes) and isinstance(boxes[0], (list, tuple)):
            boxes = boxes[0]
        if len(boxes) != 4:
            raise ValueError(f"Expected four bbox values, got {boxes}")
        return [float(v) for v in boxes]

    def _to_xyxy(self, box, width, height):
        fmt = self.bbox_format
        if fmt == "xyxy":
            x1, y1, x2, y2 = box
        elif fmt == "xywh":
            x, y, w, h = box
            x1, y1, x2, y2 = x, y, x + w, y + h
        elif fmt == "yolo":
            cx, cy, w, h = box
            x1, y1 = (cx-w/2)*width, (cy-h/2)*height
            x2, y2 = (cx+w/2)*width, (cy+h/2)*height
        else:
            raise ValueError(f"Unsupported bbox_format: {fmt}")
        x1 = max(0.0, min(width-1.0, x1)); y1 = max(0.0, min(height-1.0, y1))
        x2 = max(x1+1.0, min(float(width), x2)); y2 = max(y1+1.0, min(float(height), y2))
        return x1, y1, x2, y2

    @staticmethod
    def _xyxy_to_yolo(box, width, height):
        x1, y1, x2, y2 = box
        return [((x1+x2)/2)/width, ((y1+y2)/2)/height, (x2-x1)/width, (y2-y1)/height]

    def _expand_box(self, box, width, height):
        x1, y1, x2, y2 = box
        cx, cy = (x1+x2)/2, (y1+y2)/2
        bw = max(x2-x1, float(self.min_crop_size))*self.crop_scale
        bh = max(y2-y1, float(self.min_crop_size))*self.crop_scale
        return (max(0, int(round(cx-bw/2))), max(0, int(round(cy-bh/2))),
                min(width, int(round(cx+bw/2))), min(height, int(round(cy+bh/2))))

    def _rgb_mask_tensor(self, image, mask):
        image = TF.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)
        image = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        mask = TF.to_tensor(mask)[:1]
        return torch.cat([image, mask], dim=0)

    def _get_anatomy_ids(self, ex):
        if self.anatomy_source == "old":
            vals = list(ex.get("anatomy_ids", []))
        elif self.anatomy_source == "rough":
            vals = list(ex.get("rough_anatomy_ids", []))
            if not vals:
                rid = int(ex.get("rough_anatomy_id", 0) or 0)
                vals = [rid] if rid > 0 else []
        elif self.anatomy_source == "none":
            vals = []
        else:
            raise ValueError("anatomy_source must be old, rough, or none")

        if self.include_lesion_type:
            lesion_vals = ex.get("lesion_type_ids")
            if lesion_vals is None:
                lesion_vals = [ex.get("lesion_type_id", 0)]
            elif not isinstance(lesion_vals, list):
                lesion_vals = [lesion_vals]
            vals.extend(lesion_vals)

        cleaned, seen = [], set()
        for v in vals:
            try: v = int(v)
            except Exception: continue
            if v > 0 and v not in seen:
                cleaned.append(v); seen.add(v)
        return cleaned[:self.max_anatomy]

    def _get_anatomy_names(self, ex):
        """
        Build text conditioning from the richest available anatomy and
        lesion-type text fields.

        Preferred anatomy:
          anatomy_text
          merged_anatomy_text
          rough_anatomy_name(s)

        Preferred lesion type:
          lesion_type_text
          lesion_type_name
          lesion_type_merged
          lesion_type
        """
        names = []

        # Prefer detailed anatomy text already prepared in the annotation.
        anatomy_text = (
            ex.get("anatomy_text")
            or ex.get("merged_anatomy_text")
            or ""
        )

        if anatomy_text:
            names.append(str(anatomy_text))
        else:
            raw = ex.get(
                "rough_anatomy_names",
                ex.get("rough_anatomy_name", []),
            )

            if isinstance(raw, str):
                raw = [raw]

            names.extend(raw or [])

            if not names:
                ids = ex.get("rough_anatomy_ids", [])

                if not ids and ex.get("rough_anatomy_id"):
                    ids = [ex["rough_anatomy_id"]]

                names.extend(
                    ROUGH_ANATOMY_ID_TO_NAME.get(int(i), "")
                    for i in ids
                )

        if self.include_lesion_type:
            lesion_text = (
                ex.get("lesion_type_text")
                or ex.get("lesion_type_name")
                or ex.get("lesion_type_merged")
                or ex.get("lesion_type")
                or ""
            )

            if lesion_text:
                names.append(str(lesion_text))

        out = []
        seen = set()

        for name in names:
            # Keep phrases, but normalize formatting.
            name = str(name).replace("_", " ").strip().lower()

            if name and name not in seen:
                out.append(name)
                seen.add(name)

        return out

    def _encode_text(self, ex):
        text = " ".join(self._get_anatomy_names(ex))
        if not text: return []
        text = self.tokenizer.clean_report(text)
        return [self.tokenizer.get_id_by_token(t) for t in text.split()][:self.max_anatomy]

    def __getitem__(self, idx):
        ex = self.examples[idx]
        paths = ex["image_path"]
        if isinstance(paths, str): paths = [paths]

        if self.use_dam:
            image = self._load_rgb(paths[0])
            width, height = image.size
            xyxy = self._to_xyxy(self._first_box(ex), width, height)
            mask_path = ex.get("mask_path")
            if mask_path:
                mask = Image.open(self._resolve(mask_path)).convert("L")
                if mask.size != image.size:
                    raise ValueError(f"Image/mask mismatch for {ex['id']}: {image.size} vs {mask.size}")
                mask = mask.point(lambda p: 255 if p > 0 else 0, mode="L")
            else:
                mask = Image.new("L", (width, height), 0)
                x1,y1,x2,y2 = xyxy
                ImageDraw.Draw(mask).rectangle([int(x1), int(y1), max(int(x1), int(x2)-1), max(int(y1), int(y2)-1)], fill=255)
            crop = self._expand_box(xyxy, width, height)
            focal_image, focal_mask = image.crop(crop), mask.crop(crop)
            if self.split == "train" and random.random() < self.train_flip_prob:
                image, mask = TF.hflip(image), TF.hflip(mask)
                focal_image, focal_mask = TF.hflip(focal_image), TF.hflip(focal_mask)
            images = torch.stack([self._rgb_mask_tensor(image, mask), self._rgb_mask_tensor(focal_image, focal_mask)], 0)
            norm_boxes = [self._xyxy_to_yolo(xyxy, width, height)]
        else:
            a = self._load_rgb(paths[0]); b = self._load_rgb(paths[1]) if len(paths) > 1 else a.copy()
            if self.transform is not None: a, b = self.transform(a), self.transform(b)
            images = torch.stack([a,b], 0)
            # Legacy files generally store normalized cx,cy,w,h.
            norm_boxes = ex.get("bboxes", [])

        bbox_tensor = torch.zeros(self.max_boxes, 4, dtype=torch.float32)
        bbox_mask = torch.zeros(self.max_boxes, dtype=torch.bool)
        for i, box in enumerate(norm_boxes[:self.max_boxes]):
            bbox_tensor[i] = torch.tensor(box, dtype=torch.float32)
            bbox_mask[i] = True

        anatomy_tensor = torch.zeros(self.max_anatomy, dtype=torch.long)
        vals = self._encode_text(ex) if self.anatomy_encoding == "text" else self._get_anatomy_ids(ex)
        for i, v in enumerate(vals[:self.max_anatomy]): anatomy_tensor[i] = int(v)

        return (ex["id"], images, ex["ids"], ex["mask"], len(ex["ids"]),
                bbox_tensor, bbox_mask, anatomy_tensor)
