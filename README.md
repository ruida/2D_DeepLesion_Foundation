# 2D DeepLesion Foundation

This repository provides a complete 2D DeepLesion analysis pipeline covering:

1. **Lesion detection** with YOLO-TLP-MOE  
2. **Lesion segmentation** with Swin-UMamba ROI crop segmentation  
3. **Short radiology report generation** with R2Gen-Mamba-DAM-Anatomy-Lesion  

The pipeline is designed around 2D DeepLesion slice images. The detection model first localizes candidate lesions, the segmentation model refines the lesion mask inside detected ROIs, and the report generation model produces short lesion-centered radiology descriptions.

---

## Repository Structure

```text
2D_DeepLesion_Foundation/
├── YOLO-TLP-MOE/
│   └── README.md
├── Swin-UMamba-DeepLesion-Clean/
│   └── README.md
├── R2Gen-Mamba-DAM-Anatomy-Lesion/
│   └── README.md
└── README.md
```

---

## Overall Workflow

```text
DeepLesion 2D images
        │
        ▼
[1] YOLO-TLP-MOE Detection
        │
        ├── train lesion detector
        ├── predict YOLO bbox txt files
        └── export top-1 / top-2 candidate boxes
        │
        ▼
[2] Swin-UMamba Segmentation
        │
        ├── crop ROIs from YOLO predicted boxes
        ├── segment lesion masks in ROI space
        ├── stitch ROI masks back to original 512×512 image space
        └── evaluate Dice / IoU / Precision / Recall
        │
        ▼
[3] R2Gen-Mamba-DAM Short Report Generation
        │
        ├── use image features, lesion/ROI information, and anatomy/lesion tokens
        ├── train short-report generation model
        └── evaluate generated radiology descriptions
```

---

# 1. Detection: YOLO-TLP-MOE

Directory:

```bash
cd YOLO-TLP-MOE
```

Detailed instructions:

[YOLO-TLP-MOE README](YOLO-TLP-MOE/README.md)

## Purpose

YOLO-TLP-MOE performs 2D tiny lesion detection on DeepLesion images. It predicts lesion bounding boxes in YOLO format.

## Input

```text
VisDroneDeepLesion_clear/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── labels/
├── VisDrone2019-DET-val/
│   ├── images/
│   └── labels/
└── VisDrone2019-DET-test-dev/
    ├── images/
    └── labels/
```

Each YOLO label file contains:

```text
class_id x_center y_center width height
```

## Main Output

```text
YOLO predicted bbox txt files
```

Example:

```text
YOLO-TLP-MOE/yoloTLP_runs/DeepLesion_tlp_esmoe_p54_testdev_pred/labels/
```

These predicted boxes are passed to the segmentation stage.

## Minimal Commands

```bash
cd YOLO-TLP-MOE

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate yolo_tlp_moe

ml CUDA/12.1
ml gcc/11.3.0
```

Train:

```bash
yolo detect train \
  model=ultralytics/cfg/models/v8/yoloTLP_esmoe_p5.yaml \
  data=data/VisDroneDeepLesion_clear.yaml \
  imgsz=768 \
  batch=8 \
  epochs=300 \
  workers=8 \
  device=0 \
  project=yoloTLP_runs \
  name=DeepLesion_tlp_esmoe_p54
```

Predict:

```bash
yolo detect predict \
  model=yoloTLP_runs/DeepLesion_tlp_esmoe_p54/weights/best.pt \
  source=/data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear/VisDrone2019-DET-test-dev/images \
  imgsz=768 \
  conf=0.05 \
  iou=0.7 \
  save=True \
  save_txt=True \
  save_conf=True \
  device=0 \
  project=yoloTLP_runs \
  name=DeepLesion_tlp_esmoe_p54_testdev_pred
```

---

# 2. Segmentation: Swin-UMamba DeepLesion ROI Crop Segmentation

Directory:

```bash
cd Swin-UMamba-DeepLesion-Clean
```

Detailed instructions:

[Swin-UMamba DeepLesion README](Swin-UMamba-DeepLesion-Clean/README.md)

## Purpose

Swin-UMamba performs lesion segmentation in cropped ROI space and then stitches predicted ROI masks back to the original DeepLesion image space.

## Input

The segmentation stage uses two types of input:

### A. Training input

```text
SAM-filtered DeepLesion images and masks
```

Example:

```text
/data/ruida/SAM/sam3/data/images_filtered
/data/ruida/SAM/sam3/data/labels_filtered
```

### B. Testing input

```text
YOLO predicted bbox txt files from YOLO-TLP-MOE
```

Example:

```text
YOLO-TLP-MOE/yoloTLP_runs/DeepLesion_tlp_esmoe_p54_testdev_pred/labels/
```

## Main Output

```text
Full-size 512×512 predicted lesion masks
```

Example:

```text
Swin-UMamba-DeepLesion-Clean/roi_outputs_top1/full_masks/
Swin-UMamba-DeepLesion-Clean/roi_outputs_top2/full_masks/
```

## Minimal Commands

```bash
cd Swin-UMamba-DeepLesion-Clean

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate acl_umamba

ml CUDA/12.1
ml gcc/11.3.0
```

Set nnU-Net paths:

```bash
export nnUNet_raw=/data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_raw
export nnUNet_preprocessed=/data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_preprocessed
export nnUNet_results=/data/ruida/segmentation/Swin-UMamba-Origin/results
```

Create SAM filtered split by VisDrone/YOLO split:

```bash
python make_sam_filtered_split_by_visdrone.py \
  --sam_images_dir /data/ruida/SAM/sam3/data/images_filtered \
  --sam_masks_dir /data/ruida/SAM/sam3/data/labels_filtered \
  --visdrone_root /data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear \
  --out_root data/sam_filtered_by_visdrone_split \
  --mode symlink
```

Create Dataset711 ROI crop training dataset:

```bash
bash make_dataset711_deeplesion_crop.sh
```

Preprocess:

```bash
nnUNetv2_plan_and_preprocess -d 711 --verify_dataset_integrity
```

Train:

```bash
sbatch train_2D_DL_crop.slurm
```

Create top-1 and top-2 ROI crops from YOLO predictions:

```bash
bash run_yolo_txt_roi_swinumamba_infer_top1.sh
bash run_yolo_txt_roi_swinumamba_infer_top2.sh
```

Stitch ROI masks back to full image space:

```bash
python stitch_roi_masks.py \
  --mapping_json roi_outputs_top1/mapping.json \
  --pred_dir roi_outputs_top1/preds \
  --out_dir roi_outputs_top1/full_masks \
  --binarize

python stitch_roi_masks.py \
  --mapping_json roi_outputs_top2/mapping.json \
  --pred_dir roi_outputs_top2/preds \
  --out_dir roi_outputs_top2/full_masks \
  --binarize
```

Evaluate:

```bash
python eval_png_masks.py \
  --pred_dir roi_outputs_top1/full_masks \
  --gt_dir data/sam_filtered_by_visdrone_split/test-dev/masks \
  --out_dir eval_png_out/top1

python eval_png_masks.py \
  --pred_dir roi_outputs_top2/full_masks \
  --gt_dir data/sam_filtered_by_visdrone_split/test-dev/masks \
  --out_dir eval_png_out/top2
```

---

# 3. Short Report Generation: R2Gen-Mamba-DAM-Anatomy-Lesion

Directory:

```bash
cd R2Gen-Mamba-DAM-Anatomy-Lesion
```

Detailed instructions:

[R2Gen-Mamba-DAM-Anatomy-Lesion README](R2Gen-Mamba-DAM-Anatomy-Lesion/README.md)

## Purpose

R2Gen-Mamba-DAM-Anatomy-Lesion generates short DeepLesion-style radiology reports using image information, lesion/anatomy information, and lesion-centered textual tokens.

## Input

The report generation stage may use:

```text
DeepLesion images
lesion bounding boxes
anatomy text tokens
lesion type tokens
short report annotations
```

Example annotation fields include:

```text
image_id
report
bbox
anatomy_text
lesion_type_text
```

## Main Output

```text
Generated short lesion-centered radiology reports
```

## Minimal Commands

```bash
cd R2Gen-Mamba-DAM-Anatomy-Lesion

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate r2gen_mamba

ml CUDA/12.1
ml gcc/11.3.0
```

Train with simplified anatomy/lesion text setup:

```bash
bash scripts_text/train_totalseg_oracle_space_anatomy_text.sh
```

Evaluate or test using the corresponding testing script described in:

[R2Gen-Mamba-DAM-Anatomy-Lesion README](R2Gen-Mamba-DAM-Anatomy-Lesion/README.md)

---

## Recommended Running Order

Run the repositories in this order:

```text
1. YOLO-TLP-MOE
   Detection training and prediction
   Output: predicted YOLO bbox txt files

2. Swin-UMamba-DeepLesion-Clean
   ROI crop segmentation and stitching
   Output: full-size predicted lesion masks

3. R2Gen-Mamba-DAM-Anatomy-Lesion
   Short report generation
   Output: generated lesion-centered reports
```

---

## Data Flow Between Modules

```text
YOLO-TLP-MOE
    output:
        predicted bbox txt files
        class_id x_center y_center width height confidence
    used by:
        Swin-UMamba-DeepLesion-Clean

Swin-UMamba-DeepLesion-Clean
    output:
        ROI crop mapping JSON
        stitched full-size lesion masks
        segmentation metrics
    can be used by:
        downstream lesion analysis or report generation

R2Gen-Mamba-DAM-Anatomy-Lesion
    input:
        image features
        bbox / lesion / anatomy tokens
        report annotations
    output:
        generated short reports
```

---

## Important Notes

- Large datasets, trained weights, prediction folders, logs, and SLURM outputs should not be committed to GitHub.
- Keep only source code, scripts, README files, and lightweight configuration files in the repository.
- The `_000` suffix in DeepLesion filenames may refer to the same slice as the non-`_000` name. When matching files across detection, segmentation, and report-generation stages, use canonical filename matching when needed.
- The segmentation stage saves crop coordinates in `mapping.json`. These coordinates are used to stitch ROI predictions back into the original 512×512 image space.
- For reproducibility, report the exact model checkpoint, dataset split, confidence threshold, IoU threshold, and whether top-1 or top-2 YOLO boxes were used.

---

## Citation and Acknowledgement

This codebase builds on YOLO-style object detection, Swin-UMamba / nnU-Net style segmentation, and R2Gen-Mamba style radiology report generation. Please cite the original methods and datasets used in your experiments.
