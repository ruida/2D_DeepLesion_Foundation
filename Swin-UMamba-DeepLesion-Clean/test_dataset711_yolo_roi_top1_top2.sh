#!/bin/bash
set -euo pipefail

cd /vf/users/ruida/segmentation/Swin-UMamba-DeepLesion-Clean
mkdir -p logs

source /data/ruida/conda2/etc/profile.d/conda.sh

set +u
conda activate acl_umamba
set -u

ml CUDA/12.1
ml gcc/11.3.0

export nnUNet_raw=/vf/users/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_raw
export nnUNet_preprocessed=/vf/users/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_preprocessed
export nnUNet_results=/vf/users/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_results

echo "============================================================"
echo "Testing Dataset711_DeepLesionCrop with YOLO ROI crops"
echo "Repo: $(pwd)"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"
echo "============================================================"

echo
echo "[Check checkpoint]"
CKPT=${nnUNet_results}/Dataset711_DeepLesionCrop/nnUNetTrainerSwinUMamba__nnUNetPlans__2d/fold_all/checkpoint_best.pth
ls -lh "${CKPT}"

echo
echo "[Check ROI crops]"
echo "Top-1 input crops:"
find roi_outputs_top1/imagesTs -name "*.png" | wc -l

echo "Top-2 input crops:"
find roi_outputs_top2/imagesTs -name "*.png" | wc -l

echo
echo "[Top-1 prediction]"
rm -rf roi_outputs_top1/preds
mkdir -p roi_outputs_top1/preds

nnUNetv2_predict \
  -i roi_outputs_top1/imagesTs \
  -o roi_outputs_top1/preds \
  -d 711 \
  -c 2d \
  -f all \
  -tr nnUNetTrainerSwinUMamba \
  -chk checkpoint_best.pth

echo
echo "[Top-2 prediction]"
rm -rf roi_outputs_top2/preds
mkdir -p roi_outputs_top2/preds

nnUNetv2_predict \
  -i roi_outputs_top2/imagesTs \
  -o roi_outputs_top2/preds \
  -d 711 \
  -c 2d \
  -f all \
  -tr nnUNetTrainerSwinUMamba \
  -chk checkpoint_best.pth

echo
echo "Prediction finished."
echo "Top-1 predicted ROI masks:"
find roi_outputs_top1/preds -name "*.png" | wc -l

echo "Top-2 predicted ROI masks:"
find roi_outputs_top2/preds -name "*.png" | wc -l
