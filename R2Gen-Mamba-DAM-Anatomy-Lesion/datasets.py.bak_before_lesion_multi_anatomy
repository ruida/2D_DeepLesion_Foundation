import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class DeepLesionDataset(BaseDataset):
    """
    Dual-image dataset for DeepLesion.

    JSON image_path values are relative paths from IMAGE_DIR:
      image_path[0]: "VisDrone2019-DET-train/images/000002_02_01_050.png"
      image_path[1]: "VisDrone2019-DET-train/images/000002_02_01_050_000.png"

    Full path = IMAGE_DIR / image_path[i]
      e.g. /data/.../VisDroneDeepLesion_clear/VisDrone2019-DET-train/images/000002_02_01_050.png

    Works for all splits:
      train -> VisDrone2019-DET-train/images/
      val   -> VisDrone2019-DET-val/images/
      test  -> VisDrone2019-DET-test-dev/images/

    Both images are loaded and stacked as (2, C, H, W).
    For 11 entries with only one image path, that image is used for both slots.
    """

    MAX_BOXES   = 2
    MAX_ANATOMY = 20

    def __init__(self, args, tokenizer, split, transform=None):
        super().__init__(args, tokenizer, split, transform)
        self.max_boxes   = getattr(args, 'max_boxes',   self.MAX_BOXES)
        self.max_anatomy = getattr(args, 'max_anatomy', self.MAX_ANATOMY)

        # Anatomy source options:
        #   old   = original anatomy_ids, leakage-prone
        #   rough = single coarse rough_anatomy_id, safer
        #   none  = no anatomy token
        self.anatomy_source = getattr(args, 'anatomy_source', 'old')

    def _load(self, rel_path):
        """
        rel_path comes directly from the JSON, e.g.:
            "VisDrone2019-DET-train/images/000002_02_01_050.png"
        Full path = image_dir / rel_path  (no stripping)
        """
        full = os.path.join(self.image_dir, rel_path)
        if not os.path.isfile(full):
            raise FileNotFoundError(
                f"\n[DeepLesionDataset] Image not found: {full}\n"
                f"  rel_path  = {rel_path}\n"
                f"  image_dir = {self.image_dir}\n"
                f"  Expected: image_dir/VisDrone2019-DET-train/images/*.png"
            )
        img = Image.open(full).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        ex    = self.examples[idx]
        paths = ex['image_path']

        img_0 = self._load(paths[0])
        img_1 = self._load(paths[1]) if len(paths) >= 2 else img_0
        image = torch.stack((img_0, img_1), 0)

        report_ids   = ex['ids']
        report_masks = ex['mask']
        seq_length   = len(report_ids)

        bboxes      = ex.get('bboxes', [])[:self.max_boxes]
        bbox_tensor = torch.zeros(self.max_boxes, 4, dtype=torch.float32)
        bbox_mask   = torch.zeros(self.max_boxes,    dtype=torch.bool)
        for i, b in enumerate(bboxes):
            bbox_tensor[i] = torch.tensor(b, dtype=torch.float32)
            bbox_mask[i]   = True

        # ---------------- anatomy input ----------------
        # old:
        #   Uses original anatomy_ids from deeplesion_mamba_final.json.
        #   WARNING: this is leakage-prone because it encodes report terms.
        #
        # rough:
        #   Uses one coarse rough_anatomy_id from
        #   deeplesion_mamba_final_rough_anatomy.json.
        #
        # none:
        #   Uses all-zero anatomy tensor, effectively disabling anatomy tokens.
        anatomy_tensor = torch.zeros(self.max_anatomy, dtype=torch.long)

        if self.anatomy_source == 'old':
            anat_ids = ex.get('anatomy_ids', [])[:self.max_anatomy]
        elif self.anatomy_source == 'rough':
            rough_id = int(ex.get('rough_anatomy_id', 0))
            anat_ids = [rough_id] if rough_id > 0 else []
        elif self.anatomy_source == 'none':
            anat_ids = []
        else:
            raise ValueError(
                f"Unknown anatomy_source={self.anatomy_source}. "
                "Use one of: old, rough, none."
            )

        for i, aid in enumerate(anat_ids[:self.max_anatomy]):
            anatomy_tensor[i] = int(aid)

        return (
            ex['id'],
            image,
            report_ids,
            report_masks,
            seq_length,
            bbox_tensor,
            bbox_mask,
            anatomy_tensor,
        )
