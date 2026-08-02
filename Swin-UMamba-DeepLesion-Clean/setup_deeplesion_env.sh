#!/bin/bash

cd /data/ruida/segmentation/Swin-UMamba-DeepLesion-Clean

source /data/ruida/conda2/etc/profile.d/conda.sh
conda activate acl_umamba

ml CUDA/12.1
ml gcc/11.3.0

export nnUNet_raw=/data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_raw
export nnUNet_preprocessed=/data/ruida/segmentation/Swin-UMamba-Origin/data/nnUNet_preprocessed
export nnUNet_results=/data/ruida/segmentation/Swin-UMamba-Origin/results

echo "Environment ready"
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"
which python
python --version
