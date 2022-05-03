# Improved 3D abdominal multi-organ segmentation via boundary-constrained models

Pytorch Code for the paper "Improved 3D abdominal multi-organ segmentation via boundary-constrained models"

This repository contains pytorch implementation corresponding to single-task (baseline) and multi-task (boundary-constrained) models.

## 3D baseline models:
1. 3D UNet
2. 3D UNet++
3. 3D Attention-UNet

Implementation for baseline models is in script [here](https://github.com/samra-irshad/3d-multitask-unet/blob/main/model/baseline_models.py)

## 3D boundary-constrained models:
1. 3D UNet-MTL-TSOL
2. 3D UNet++-MTL-TSOL
3. 3D UNet-MTL-TSD
4. 3D UNet++-MTL-TSD
5. 3D Attention-UNet-MTL-TSD
6. 3D Attention-UNet-MTL-TSOL

Implementation for boundary-constrained models is in script [here](https://github.com/samra-irshad/3d-multitask-unet/blob/main/model/boundary_constrained_models.py)

## Model Training 
To train the baseline models, use the following command:

`python train_baseline.py --data_folder 'path where the data is stored' --output_folder 'path to save the results' --batch 4 --epochs 300 --model unet --lr 0.001`

To train the boundary-constrained models, use the following command:

`python train_3d_boundary_constrained.py --data_folder 'path where the data is stored' --output_folder 'path to save the results' --batch 4 --epochs 300 \
--model unet --conf tsol --lr 0.001 --lambda_edge 0.5`
