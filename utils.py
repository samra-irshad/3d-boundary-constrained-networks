import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import SimpleITK as sitk
import warnings
warnings.filterwarnings("ignore")
from medpy.metric.binary import dc, ravd
import sys
from skimage import measure


def getOneHotSegmentation(batch):
    
    backgroundVal = 0.

    # Chaos MRI (These values are to set label values as 0,1,2,3 and 4)
    label1 = 1.
    label2 = 2.
    label3 = 3.
    label4 = 4.
    label5 = 5.
    label6 = 6.
    label7 = 7.
    label8 = 8.


    oneHotLabels = torch.cat(
        (batch == backgroundVal, batch == label1, batch == label2, batch == label3, batch == label4, batch == label5, batch == label6, batch == label7, batch == label8),
        dim=1)

    return oneHotLabels.float()

def dice_score(outputs, labels, max_label):
    """
    Evaluation function for Dice score of segmentation overlap
    """
    dice = torch.FloatTensor(max_label-1).fill_(0).to(outputs.device)
    
    for label_num in range(1, max_label):
        iflat = (outputs==label_num).view(-1).float()
        tflat = (labels==label_num).view(-1).float()
        intersection = (iflat * tflat).sum()
        dice[label_num-1] = (2. * intersection) / (iflat.sum() + tflat.sum())
    return dice

def hd_updated(label_GT, label_CNN):
    seg = sitk.GetImageFromArray(label_CNN, isVector=False)
    reference_segmentation  = sitk.GetImageFromArray(label_GT, isVector=False)
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    #label = 1
    reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(reference_segmentation, squaredDistance=False, useImageSpacing=True))
    reference_surface = sitk.LabelContour(reference_segmentation)
    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum()) 
    hausdorff_distance_filter.Execute(reference_segmentation, seg)
    hd_new = hausdorff_distance_filter.GetAverageHausdorffDistance()
    
    segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(seg, squaredDistance=False, useImageSpacing=True))
    segmented_surface = sitk.LabelContour(seg)
        
    # Multiply the binary surface segmentations with the distance maps. The resulting distance
    # maps contain non-zero values only on the surface (they can also contain zero on the surface)
    seg2ref_distance_map = reference_distance_map*sitk.Cast(segmented_surface, sitk.sitkFloat32)
    ref2seg_distance_map = segmented_distance_map*sitk.Cast(reference_surface, sitk.sitkFloat32)
        
    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())
    
    # Get all non-zero distances and then add zero distances if required.
    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr!=0]) 
    seg2ref_distances = seg2ref_distances + \
                        list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr!=0]) 
    ref2seg_distances = ref2seg_distances + \
                        list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))
        
    all_surface_distances = seg2ref_distances + ref2seg_distances

    # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In 
    # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two 
    # segmentations, though in our case it is. More on this below.
    msd_new = np.mean(all_surface_distances)
    
    return hd_new

def avg_hd(imageDataCNN, imageDataGT, max_label):
   
    
    hd1 = np.zeros((max_label))
    for c_i in range(max_label):
        #print('class', c_i)
        label_GT = np.zeros(imageDataGT.shape)
        label_CNN = np.zeros(imageDataCNN.shape)
        idx_GT = np.where(imageDataGT == c_i+1)
        label_GT[idx_GT] = 1
        idx_CNN = np.where(imageDataCNN == c_i+1)
        label_CNN[idx_CNN] = 1
        label_CNN=label_CNN.astype('uint8')
        label_GT = label_GT.astype('uint8')
       
        if np.count_nonzero(label_CNN)>0 and np.count_nonzero(label_GT)>0:
            hd1[c_i] = hd_updated(label_GT,label_CNN)
        elif np.count_nonzero(label_GT)>0 and np.count_nonzero(label_CNN)==0:
            hd1[c_i]=2.
        elif np.count_nonzero(label_GT)==0 and np.count_nonzero(label_CNN)>0:
            hd1[c_i]=2.
        elif np.count_nonzero(label_GT)==0 and np.count_nonzero(label_CNN)==0:          
            hd1[c_i]=0.

    return hd1

def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class diceloss(nn.Module):
    """
    The Dice Loss function
    """
    def __init__(self, smooth=1e-6):
        super(diceloss, self).__init__()
        self.smooth = smooth

    def forward(self, probs, labels):
        numerator = 2 * torch.sum(labels * probs, (2,3,4))
        denominator = torch.sum(labels + probs**2, (2,3,4)) + self.smooth                                       
             
        return 1 - torch.mean(numerator/denominator)

class CE_loss(nn.Module):
    def __init__(self):
        
        super(CE_loss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
       
    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """      
        ce_loss = self.ce(net_output, target.long())            
        return ce_loss

class DC_CE_loss(nn.Module):
    def __init__(self, weight_ce=1, weight_dice=1):
        super(DC_CE_loss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ce = nn.CrossEntropyLoss()   
        self.dc = diceloss()
       
    def forward(self, net_output, target_ce, target_dc):        
  
        dc_loss = self.dc(net_output, target_dc) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target.long()) if self.weight_ce != 0 else 0
        seg_loss =  ce_loss + dc_loss
              
        return seg_loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def augmentAffine(img_in, seg_in, strength=0.05):
    """
    3D affine augmentation on image and segmentation mini-batch on GPU.
    (affine transf. is centered: trilinear interpolation and zero-padding used for sampling)
    :input: img_in batch (torch.cuda.FloatTensor), seg_in batch (torch.cuda.LongTensor)
    :return: augmented BxCxTxHxW image batch (torch.cuda.FloatTensor), augmented BxTxHxW seg batch (torch.cuda.LongTensor)
    """
    B,C,D,H,W = img_in.size()
    
    affine_matrix = (torch.eye(3,4).unsqueeze(0) + torch.randn(B, 3, 4) * strength).to(img_in.device)
    meshgrid = F.affine_grid(affine_matrix,torch.Size((B,1,D,H,W)))
    img_out = F.grid_sample(img_in, meshgrid,padding_mode='border')
    seg_out = F.grid_sample(seg_in.float().unsqueeze(1), meshgrid, mode='nearest').long().squeeze(1)
    return img_out, seg_out

def dice_score(outputs, labels, max_label):
    """
    Evaluation function for Dice score of segmentation overlap
    """
    dice = torch.FloatTensor(max_label-1).fill_(0).to(outputs.device)
    
    for label_num in range(1, max_label):
        iflat = (outputs==label_num).view(-1).float()
        tflat = (labels==label_num).view(-1).float()
        intersection = (iflat * tflat).sum()
        dice[label_num-1] = (2. * intersection) / (iflat.sum() + tflat.sum())
    return dice
