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

## Downloading the datasets:

First download the datasets from links below:
1. Pancreas-CT dataset {For images, Use [Link1](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT) , For labels: Use [Link2](https://zenodo.org/record/1169361#.YnIytuhBw2w)}
2. BTCV dataset {For images: Use [Link1](https://www.synapse.org/#!Synapse:syn3193805) , For labels: Use [Link2](https://zenodo.org/record/1169361#.YnIytuhBw2w)}

## Dataset preparation:
1. To prepare the data, we use the pipeline utilized in [Obelisk-Net](https://www.sciencedirect.com/science/article/abs/pii/S136184151830611X) paper
2. Use the ITK-snap to crop the scans and labels according to the bounding box coordinates given on [Link2](https://zenodo.org/record/1169361#.YnIytuhBw2w).

## Dataset organization
### Data organization for training baseline models:
Organize the CT scans and their corresponding labels according to the format below:
```
Data Folder:
     --data:
            --images1:
                     --pancreas_ct1.nii.gz
                     --pancreas_ct2.nii.gz
                     .....................
            --labels1:
                     --label_ct1.nii.gz
                     --label_ct2.nii.gz
                     .....................
```
### Data organization for training boundary-constrained models:
Organize the CT scans and their corresponding labels according to the format below:
```
Data Folder:
     --data:
            --images1:
                     --pancreas_ct1.nii.gz
                     --pancreas_ct2.nii.gz
                     .....................
            --labels1:
                     --label_ct1.nii.gz
                     --label_ct2.nii.gz
                     .....................
            --contours:
                     --label_ct1.nii.gz
                     --label_ct2.nii.gz
                     .....................
```
## Model Training 
To train the baseline models, use the following command:

`python train_baseline.py --data_folder 'path where the data is stored' --output_folder 'path to save the results' --batch 4 --epochs 300 --model unet --lr 0.001`

To train the boundary-constrained models, use the following command:

`python train_3d_boundary_constrained.py --data_folder 'path where the data is stored' --output_folder 'path to save the results' --batch 4 --epochs 300 \
--model unet --conf tsol --lr 0.001 --lambda_edge 0.5`

The Models have been trained using two P100 GPUs, you will need to reduce the size of the batch if you use one GPU. 

## Model training on custom dataset
To train the baseline models on a different dataset, organize the data in the format described above and then use the training commands.
