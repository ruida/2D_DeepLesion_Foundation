#!/bin/bash


cd /data/ruida/segmentation/Swin-UMamba-DeepLesion-Clean

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate acl_umamba

ml CUDA/12.1
ml gcc/11.3.0

export nnUNet_raw=/data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_raw
export nnUNet_preprocessed=/data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_preprocessed
export nnUNet_results=/data/ruida/segmentation/Swin-UMamba-Origin/results

BASE=data/sam_filtered_by_visdrone_split

python roi_tight_cropping_scripts/roi_scripts/make_roi_dataset_tight.py \
  --dataset_id 711 \
  --dataset_name DeepLesionCrop \
  --images_dir ${BASE}/train/images \
  --labels_dir ${BASE}/train/masks \
  --report_json /data/ruida/SAM/sam3/data/report.json \
  --nnunet_raw $nnUNet_raw \
  --split train \
  --margin 16 \
  --min_box_size 8 \
  --max_rois_per_image 1

echo "Done creating Dataset711 from SAM train split."
echo "Output:"
echo "$nnUNet_raw/Dataset711_DeepLesionCrop"
