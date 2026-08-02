# DeepLesion Crop Segmentation with Swin-UMamba

This repository adapts **Swin-UMamba / nnU-Net v2** for DeepLesion 2D lesion segmentation using a crop-and-stitch pipeline.

The main idea is:

```text
DeepLesion full 512x512 image
        ↓
lesion ROI box
        ↓
crop image patch around the lesion
        ↓
Swin-UMamba predicts lesion mask on the crop
        ↓
stitch / paste the crop mask back into original image space
        ↓
full-size 512x512 predicted lesion mask
        ↓
evaluate against DeepLesion ground-truth mask
```

There are two modes:

1. **Training / validation mode**: use ground-truth DeepLesion boxes from `report.json` to make ROI crop images and crop masks.
2. **Realistic testing mode**: use YOLO-TLP-MOE predicted boxes, segment each predicted ROI crop with Swin-UMamba, then stitch the predicted ROI masks back to the original DeepLesion image space.

---

## 1. Repository layout

Important files added for the DeepLesion crop segmentation pipeline:

```text
Swin-UMamba-Origin/
├── README.md                                      # original Swin-UMamba README
├── make_roi_inference_crops_from_yolo_txt.py      # YOLO txt → nnUNet imagesTs ROI crops
├── stitch_roi_masks.py                            # ROI predictions → full-size masks
├── eval_png_masks.py                              # PNG mask evaluation
├── match_top1_predictions_to_dataset719_gt.py     # optional GT matching helper
├── run_dataset719_yolo_roi_swinumamba_top1_foldall.sh
├── run_dataset719_yolo_roi_swinumamba_top2_foldall.sh
├── train_2D_DL_crop.slurm                         # train ROI crop model, Dataset711
├── test_2D_DL_crop.slurm                          # predict ROI crop masks, Dataset711
├── pre_deeplesion.slurm                           # preprocess full-image dataset, Dataset717
├── pre_2D_DL716_DeepLesion3D.slurm                # preprocess Dataset716 if used
├── roi_scripts/
│   ├── make_roi_dataset_tight.py                  # GT box → ROI training dataset
│   ├── make_roi_inference_crops_tight.py          # JSON boxes → ROI inference crops
│   ├── stitch_roi_masks.py                        # older stitch script
│   └── detect_yolo_boxes.py                       # optional detector wrapper
└── swin_umamba/                                   # nnU-Net v2 / Swin-UMamba package
```

For release, keep one canonical version of each script in the root or under `roi_scripts/`. The recommended release entry points are:

```text
roi_scripts/make_roi_dataset_tight.py
make_roi_inference_crops_from_yolo_txt.py
stitch_roi_masks.py
eval_png_masks.py
run_dataset719_yolo_roi_swinumamba_top2_foldall.sh
```

---

## 2. Environment setup

Create and activate the environment:

```bash
conda create -n acl_umamba python=3.10 -y
conda activate acl_umamba
```

Install PyTorch and Swin-UMamba dependencies. Adjust CUDA/PyTorch versions for your cluster.

```bash
pip install torch torchvision
pip install causal-conv1d mamba-ssm
pip install torchinfo timm numba pillow numpy scipy pandas scikit-image
```

Install the Swin-UMamba / nnU-Net v2 package in editable mode:

```bash
cd /data/ruida/segmentation/Swin-UMamba-Origin/swin_umamba
pip install -e .
```

On the NIH/cluster environment used in this project:

```bash
source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate acl_umamba
ml CUDA/12.1
ml gcc/11.3
```

---

## 3. Set nnU-Net paths

Before preprocessing, training, or inference, export these paths:

```bash
export PROJECT_DIR=/data/ruida/segmentation/Swin-UMamba-Origin

export nnUNet_raw=${PROJECT_DIR}/data/nnUNet_raw
export nnUNet_preprocessed=${PROJECT_DIR}/data/nnUNet_preprocessed
export nnUNet_results=${PROJECT_DIR}/data/nnUNet_results

cd ${PROJECT_DIR}
```

The examples below use this project root:

```text
/data/ruida/segmentation/Swin-UMamba-Origin
```

---

## 4. Input data expected by the crop pipeline

### 4.1 DeepLesion image and mask folders

For GT-box crop training, the original helper script expects:

```text
/data/ruida/SAM/sam3/data/images_filtered/*.png
/data/ruida/SAM/sam3/data/labels_filtered/*.png
/data/ruida/SAM/sam3/data/report.json
```

`report.json` must contain lesion entries with the field:

```text
Bounding_boxes
```

The bounding box format is:

```text
x1, y1, x2, y2
```

### 4.2 YOLO-TLP-MOE predicted boxes for realistic testing

For realistic testing, YOLO-TLP-MOE must already have produced one `.txt` file per image.

The expected YOLO txt format is:

```text
class x_center y_center width height confidence
```

The coordinates are normalized YOLO coordinates. Example:

```text
0 0.241736 0.508267 0.0727046 0.0834989 0.313698
```

Typical test input folders:

```text
${PROJECT_DIR}/dataset719_yolo_test/images
${PROJECT_DIR}/dataset719_yolo_test/yolotlpmoe_predictions/labels
```

---

## 5. Training pipeline: GT boxes → ROI crop dataset → Swin-UMamba

Training uses GT lesion boxes to create tight ROI crops. This isolates the segmentation task from the detector during training.

### Step 5.1 Create the ROI crop nnU-Net dataset

This creates `Dataset711_DeepLesionROI` under `${nnUNet_raw}`.

```bash
python roi_scripts/make_roi_dataset_tight.py \
  --dataset_id 711 \
  --dataset_name DeepLesionROI \
  --images_dir /data/ruida/SAM/sam3/data/images_filtered \
  --labels_dir /data/ruida/SAM/sam3/data/labels_filtered \
  --report_json /data/ruida/SAM/sam3/data/report.json \
  --nnunet_raw ${nnUNet_raw} \
  --split train \
  --margin 16 \
  --min_box_size 8
```

The script writes:

```text
${nnUNet_raw}/Dataset711_DeepLesionROI/imagesTr/*_0000.png
${nnUNet_raw}/Dataset711_DeepLesionROI/labelsTr/*.png
${nnUNet_raw}/Dataset711_DeepLesionROI/dataset.json
```

### Step 5.2 Plan and preprocess Dataset711

```bash
nnUNetv2_plan_and_preprocess -d 711 --verify_dataset_integrity
```

### Step 5.3 Train Swin-UMamba on ROI crops

```bash
nnUNetv2_train 711 2d all \
  -tr nnUNetTrainerSwinUMamba \
  2>&1 | tee train_2D_deeplesion_crop.log
```

The corresponding SLURM script is:

```bash
sbatch train_2D_DL_crop.slurm
```

Recommended SLURM correction before release:

```bash
# wrong
#SBATCH --job-nam="train_2D_deeplesion_crop"

# correct
#SBATCH --job-name="train_2D_deeplesion_crop"
```

---

## 6. Optional direct ROI-crop prediction

This predicts masks for images already placed in:

```text
${nnUNet_raw}/Dataset711_DeepLesionROI/imagesTs
```

Command:

```bash
nnUNetv2_predict \
  -i ${nnUNet_raw}/Dataset711_DeepLesionROI/imagesTs \
  -o ${nnUNet_results}/Dataset711_DeepLesionROI/predictionsTs \
  -d 711 \
  -c 2d \
  -f all \
  -tr nnUNetTrainerSwinUMamba \
  --save_probabilities
```

The corresponding SLURM script is:

```bash
sbatch test_2D_DL_crop.slurm
```

---

## 7. Realistic testing pipeline: YOLO predicted box → crop → Swin-UMamba → stitch

This is the main testing pipeline for code release.

It uses predicted YOLO-TLP-MOE boxes instead of GT boxes.

### Step 7.1 Prepare input folders

Expected full-size test images:

```text
${PROJECT_DIR}/dataset719_yolo_test/images/*.png
```

Expected YOLO predicted label files:

```text
${PROJECT_DIR}/dataset719_yolo_test/yolotlpmoe_predictions/labels/*.txt
```

Each `.txt` filename should match the image stem.

Example:

```text
000015_02_01_106.png
000015_02_01_106.txt
```

### Step 7.2 Generate ROI crops from YOLO predicted boxes

Top-1 ROI per image:

```bash
python make_roi_inference_crops_from_yolo_txt.py \
  --images_dir ${PROJECT_DIR}/dataset719_yolo_test/images \
  --yolo_labels_dir ${PROJECT_DIR}/dataset719_yolo_test/yolotlpmoe_predictions/labels \
  --out_imagesTs ${PROJECT_DIR}/dataset719_roi_outputs_top1_foldall/roi_infer/imagesTs \
  --mapping_json ${PROJECT_DIR}/dataset719_roi_outputs_top1_foldall/roi_infer/mapping.json \
  --conf_th 0.20 \
  --iou_nms 0.70 \
  --margin 16 \
  --min_box_size 8 \
  --max_rois_per_image 1
```

Top-2 ROIs per image:

```bash
python make_roi_inference_crops_from_yolo_txt.py \
  --images_dir ${PROJECT_DIR}/dataset719_yolo_test/images \
  --yolo_labels_dir ${PROJECT_DIR}/dataset719_yolo_test/yolotlpmoe_predictions/labels \
  --out_imagesTs ${PROJECT_DIR}/dataset719_roi_outputs_top2_foldall/roi_infer/imagesTs \
  --mapping_json ${PROJECT_DIR}/dataset719_roi_outputs_top2_foldall/roi_infer/mapping.json \
  --conf_th 0.20 \
  --iou_nms 0.70 \
  --margin 16 \
  --min_box_size 8 \
  --max_rois_per_image 2
```

Output:

```text
roi_infer/imagesTs/<case>__roi00_0000.png
roi_infer/imagesTs/<case>__roi01_0000.png
roi_infer/mapping.json
```

The `mapping.json` records the original image name, ROI crop name, original image size, confidence score, and crop coordinates `[x1, y1, x2, y2]`.

### Step 7.3 Predict ROI masks with Swin-UMamba

Top-2 example:

```bash
nnUNetv2_predict \
  -d 711 \
  -c 2d \
  -f all \
  -i ${PROJECT_DIR}/dataset719_roi_outputs_top2_foldall/roi_infer/imagesTs \
  -o ${PROJECT_DIR}/dataset719_roi_outputs_top2_foldall/roi_infer/preds \
  -tr nnUNetTrainerSwinUMamba \
  -npp 1 \
  -nps 1
```

### Step 7.4 Stitch ROI masks back to 512x512 image space

Top-2 example:

```bash
python stitch_roi_masks.py \
  --mapping_json ${PROJECT_DIR}/dataset719_roi_outputs_top2_foldall/roi_infer/mapping.json \
  --pred_dir ${PROJECT_DIR}/dataset719_roi_outputs_top2_foldall/roi_infer/preds \
  --out_dir ${PROJECT_DIR}/dataset719_roi_outputs_top2_foldall/full_masks \
  --binarize
```

Output:

```text
${PROJECT_DIR}/dataset719_roi_outputs_top2_foldall/full_masks/*.png
```

Each output PNG is a full-size binary lesion mask in the original DeepLesion image space.

### Step 7.5 Run the full wrapper script

Top-1:

```bash
bash run_dataset719_yolo_roi_swinumamba_top1_foldall.sh
```

Top-2:

```bash
bash run_dataset719_yolo_roi_swinumamba_top2_foldall.sh
```

The top-2 script performs:

```text
YOLO txt labels
  → ROI crops
  → Swin-UMamba ROI mask prediction
  → stitch back to full-size masks
```

---

## 8. Evaluation

### 8.1 Evaluate predicted full masks when filenames match GT masks

If predicted masks and GT masks have the same filename, run:

```bash
python eval_png_masks.py \
  --pred_dir ${PROJECT_DIR}/dataset719_roi_outputs_top2_foldall/full_masks \
  --gt_dir ${PROJECT_DIR}/data/nnUNet_raw/Dataset719_DeepLesion2D/labelsTs \
  --out_dir ${PROJECT_DIR}/dataset719_roi_outputs_top2_foldall/eval_png_out \
  --pred_thresh 0 \
  --gt_thresh 0
```

Outputs:

```text
eval_png_out/metrics_per_case.csv
eval_png_out/metrics_summary.txt
```

Metrics include:

```text
Dice
IoU / Jaccard
Precision
Recall
Specificity
TP / FP / FN / TN
```

### 8.2 Match GT masks by image content when filenames differ

If YOLO test image names and Dataset719 names do not directly match, use the matching helper first.

Edit these paths inside `match_top1_predictions_to_dataset719_gt.py`:

```python
YOLO_IMAGE_DIR = Path("/data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear/VisDrone2019-DET-test-dev/images")
PRED_DIR = Path("/data/ruida/segmentation/Swin-UMamba-Origin/roi_outputs_top1/full_masks")
NNUNET_IMAGE_DIR = Path("/data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_raw/Dataset719_DeepLesion2D/imagesTs")
NNUNET_GT_DIR = Path("/data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_raw/Dataset719_DeepLesion2D/labelsTs")
OUT_GT_DIR = Path("/data/ruida/segmentation/Swin-UMamba-Origin/roi_outputs_top1/gt_masks_content_matched")
OUT_MAPPING = Path("/data/ruida/segmentation/Swin-UMamba-Origin/roi_outputs_top1/content_match_mapping.json")
```

Then run:

```bash
python match_top1_predictions_to_dataset719_gt.py
```

Then evaluate:

```bash
python eval_png_masks.py \
  --pred_dir /data/ruida/segmentation/Swin-UMamba-Origin/roi_outputs_top1/full_masks \
  --gt_dir /data/ruida/segmentation/Swin-UMamba-Origin/roi_outputs_top1/gt_masks_content_matched \
  --out_dir /data/ruida/segmentation/Swin-UMamba-Origin/roi_outputs_top1/eval_png_out
```

---

## 9. Full-image baseline with Dataset719

Dataset719 is the direct full-image DeepLesion 2D segmentation dataset.

Preprocess:

```bash
nnUNetv2_plan_and_preprocess -d 719 --verify_dataset_integrity
```

Train:

```bash
nnUNetv2_train 719 2d all \
  -tr nnUNetTrainerSwinUMamba \
  2>&1 | tee train_2D_Dataset719_DeepLesion2D.log
```

Predict:

```bash
nnUNetv2_predict \
  -i ${nnUNet_raw}/Dataset719_DeepLesion2D/imagesTs \
  -o ${PROJECT_DIR}/results/Dataset719_DeepLesion2D/pred_2d \
  -d 719 \
  -c 2d \
  -tr nnUNetTrainerSwinUMamba \
  -f all
```

Evaluate:

```bash
python eval_png_masks.py \
  --pred_dir ${PROJECT_DIR}/results/Dataset719_DeepLesion2D/pred_2d \
  --gt_dir ${nnUNet_raw}/Dataset719_DeepLesion2D/labelsTs \
  --out_dir ${PROJECT_DIR}/results/Dataset719_DeepLesion2D/pred_2d_eval
```

This baseline is useful to compare full-image segmentation versus crop-and-stitch segmentation.

---

## 10. Recommended GitHub release checklist

Before pushing to GitHub, clean the repository:

```bash
find . -name "__pycache__" -type d -prune -exec rm -rf {} +
find . -name "*.pyc" -delete
```

Do not commit large generated folders:

```text
data/nnUNet_raw/
data/nnUNet_preprocessed/
data/nnUNet_results/
dataset719_roi_outputs*/
roi_outputs*/
eval_png_out/
logs/
*.pth
*.pt
*.ckpt
```

Add these to `.gitignore`:

```gitignore
__pycache__/
*.pyc
*.log
*.out
*.err

/data/nnUNet_raw/
/data/nnUNet_preprocessed/
/data/nnUNet_results/
/dataset719_roi_outputs*/
/roi_outputs*/
/eval_png_out/
/logs/

*.pth
*.pt
*.ckpt
```

Fix SLURM typo in scripts:

```text
#SBATCH --job-nam=...
```

should be:

```text
#SBATCH --job-name=...
```

Recommended final GitHub README structure:

```text
1. Overview
2. Installation
3. Data preparation
4. ROI crop training
5. Realistic YOLO-box testing
6. Stitching back to original image space
7. Evaluation
8. Reproducibility notes
9. Citation
```

---

## 11. Reproducibility notes

Important hyperparameters used in the current crop pipeline:

```text
Dataset ID for ROI crop model: 711
Trainer: nnUNetTrainerSwinUMamba
Configuration: 2d
Fold: all
YOLO confidence threshold: 0.20
YOLO NMS IoU threshold: 0.70
Crop margin: 16 pixels
Minimum crop box size: 8 pixels
Top-K ROIs per image: 1 or 2
Output mask format: PNG, binary foreground > 0
```

The realistic testing flow should be reported as:

```text
YOLO-TLP-MOE predicted bounding box
        ↓
ROI crop extraction
        ↓
Swin-UMamba ROI lesion segmentation
        ↓
ROI predicted mask
        ↓
stitch / paste ROI mask back to original 512x512 DeepLesion image space
        ↓
full-image predicted lesion mask
        ↓
Dice / IoU / Precision / Recall evaluation
```

---

## 12. Citation

If using this repository, cite the original Swin-UMamba paper:

```bibtex
@article{Swin-UMamba,
    title={Swin-UMamba: Mamba-based UNet with ImageNet-based pretraining},
    author={Jiarun Liu and Hao Yang and Hong-Yu Zhou and Yan Xi and Lequan Yu and Yizhou Yu and Yong Liang and Guangming Shi and Shaoting Zhang and Hairong Zheng and Shanshan Wang},
    journal={arXiv preprint arXiv:2402.03302},
    year={2024}
}
```

Also cite nnU-Net v2 and DeepLesion according to their official citation instructions.
