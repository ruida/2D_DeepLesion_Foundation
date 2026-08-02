cd /vf/users/ruida/segmentation/Swin-UMamba-DeepLesion-Clean

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate acl_umamba
ml CUDA/12.1
ml gcc/11.3.0

export nnUNet_raw=/vf/users/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_raw
export nnUNet_preprocessed=/vf/users/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_preprocessed
export nnUNet_results=/vf/users/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_results

python make_sam_filtered_split_by_visdrone.py \
  --sam_images_dir /data/ruida/SAM/sam3/data/images_filtered \
  --sam_masks_dir /data/ruida/SAM/sam3/data/labels_filtered \
  --visdrone_root /data/ruida/object_detection/YOLO-TLP/data/VisDroneDeepLesion_clear \
  --out_root data/sam_filtered_by_visdrone_split \
  --mode symlink

bash make_dataset711_from_sam_split_train.sh

nnUNetv2_plan_and_preprocess -d 711 --verify_dataset_integrity

sbatch train_2D_DL_crop.slurm

bash make_testdev_yolo_roi_crops_top1_top2.sh

sbatch test_dataset711_yolo_roi_top1_top2.slurm

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

python eval_png_masks_dice_iou_hd95.py \
  --pred_dir roi_outputs_top1/full_masks \
  --gt_dir data/sam_filtered_by_visdrone_split/test-dev/masks \
  --out_dir eval_png_out/top1_testdev_hd95 \
  --include_missing_pred_as_empty

python eval_png_masks_dice_iou_hd95.py \
  --pred_dir roi_outputs_top2/full_masks \
  --gt_dir data/sam_filtered_by_visdrone_split/test-dev/masks \
  --out_dir eval_png_out/top2_testdev_hd95 \
  --include_missing_pred_as_empty
