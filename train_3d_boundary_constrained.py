import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import dill
import os
import nibabel as nib
import argparse

from utils import count_parameters, diceloss, CE_loss, DC_CE_loss, dice_score, getOneHotSegmentation, avg_hd, weights_init, augmentAffine, mtl_loss

from models.models_3d_mtl import unet3d_mtl_tsol, unet3d_mtl_tsd, attenunet3d_mtl_tsol, attenunet3d_mtl_tsd, unetplus3d_mtl_tsol, unetplus3d_mtl_tsd


def main():
    parser = argparse.ArgumentParser(description='multi_task_model')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('--batch', default=1, type=int,
                    metavar='N', help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
    parser.add_argument('--data_folder', default='../data', type=str, metavar='PATH',
                    help='path --> root dataset')
    parser.add_argument('--model', default='unet', type=str,
                    help='name of model')
    parser.add_argument('--conf', default='tsol', type=str,
                    help='multi-task model configuration')
    parser.add_argument('--aug', default=False, type=str,
                    help='data augmentation')
    parser.add_argument('--lambda_edge', default=1., type=float,
                    help='boundary loss weight')
    parser.add_argument('--output_folder', default='../data', type=str,
                    help='path to output folder for saving the results')
    parser.add_argument('--optimizer', default='adam', type=str,
                    help='path to output folder for saving the results')



    args = parser.parse_args()

    ####### ---- #######
    ## CREATING OUTPUT FOLDER ##
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    imgs = []
    segs = []
    edges = []
    
    len_data = len(os.listdir(os.path.join(args.data_folder,'images1')))
    list_data = np.arange(len_data)+1

    for i in range(1, len(list_data)+1):
        filescan1 = 'pancreas_ct'+str(i)+'.nii.gz'
        img = nib.load(os.path.join(args.data_folder,'images1', filescan1)).get_data()
        fileseg1 = 'label_ct'+str(i)+'.nii.gz'
        seg = nib.load(os.path.join(args.data_folder,'labels1', fileseg1)).get_data()
        seg[seg==11]=2.
        seg[seg==14]=8.
        edge = nib.load(os.path.join(args.data_folder,'contours', fileseg1)).get_data()

        imgs.append(torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float())
        segs.append(torch.from_numpy(seg).unsqueeze(0).long())
        edges.append(torch.from_numpy(edge).unsqueeze(0))
       
    imgs = torch.cat(imgs,0)
    segs = torch.cat(segs,0)
    edges = torch.cat(edges,0)
  

    # imgs = imgs/1024.0 + 1.0 #only apply scaling for pancreas-ct dataset

    num_labels=9 ## (8 organs + 1 background)

    ### --- model initialization --- ###

    if args.model == "unet":
        if args.conf == "tsol":
            net = unet3d_mtl_tsol()
        elif args.conf == "tsd":
            net = unet3d_mtl_tsd()
    if args.model == "unetplus":
        if args.conf == "tsol":
            net = unetplus3d_mtl_tsol()
        elif args.conf == "tsd":
            net = unetplus3d_mtl_tsd()
    if args.model == "attenunet":
        if args.conf == "tsol":
            net = attenunet3d_mtl_tsol()
        elif args.conf == "tsd":
            net = attenunet3d_mtl_tsd()

    net.apply(weights_init)
    print(args.model+' params: ',count_parameters(net))
    net = nn.DataParallel(net)
    net.cuda() 

    criterion = mtl_loss()
    

    #### optimizer ####
    if args.optimizer == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    if args.optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr)
    if args.optimizer == "RMSprop":
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr)
   

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,0.99)

    run_loss = np.zeros(args.epochs)

    fold_size = imgs.size(0)
    fold_size4 = fold_size - fold_size%4
    
    high_val=0.
    ep1=0
    
    np.random.seed(1)
    idx_epoch = np.random.permutation(len_data)
    train_idx= idx_epoch[0:28]
    val_idx=idx_epoch[28:32]
    test_idx = idx_epoch[32:42]
    print('idx_epoch', idx_epoch)
    print('val_idx', val_idx)
    print('test_idx', test_idx)
    val_dc_plot=[]
    yu=[]
    seg_loss=[]
    cls_loss=[]
    

    for epoch in range(args.epochs):

        net.train()
        
        run_loss[epoch] = 0.0
        t1 = 0.0
        np.random.seed(epoch)
        idx_epoch = np.random.permutation(train_idx)
        idx_epoch = torch.from_numpy(idx_epoch).view(args.batch,-1)
        t0 = time.time()

        for iter in range(idx_epoch.size(1)):
            idx = idx_epoch[:,iter] 
            
            with torch.no_grad():
                if args.aug:
                    imgs_cuda, y_label = augmentAffine(imgs[idx,:,:,:,:].cuda(), segs[idx,:,:,:].cuda(),strength=0.075)
                    edge_label = edges[idx,:,:,:].cuda()
                else:
                    imgs_cuda, y_label, edge_label = imgs[idx,:,:,:,:].cuda(), segs[idx,:,:,:].cuda(), edges[idx,:,:,:].cuda()

                torch.cuda.empty_cache()
            y_label_dc = torch.unsqueeze(y_label, 1)
            y_label_dc =  getOneHotSegmentation(y_label_dc)
            edge_label = torch.unsqueeze(edge_label, 1)
            optimizer.zero_grad() 
            predict, edge = net(imgs_cuda)
            total_loss, s_loss, c_loss = criterion(F.softmax(predict,dim=1), y_label_dc, edge, edge_label, args.lambda_edge)
            seg_loss.append(s_loss.cpu().detach().numpy())
            cls_loss.append(c_loss.cpu().detach().numpy())

            total_loss.backward()
            run_loss[epoch] += total_loss.item()
            optimizer.step()
            del total_loss; del predict; del imgs_cuda; del y_label_dc
            torch.cuda.empty_cache()
        scheduler.step()
        
        #evaluation on validation images
        t1 = time.time()-t0
        print('epoch',epoch,'time train','%.3f'%t1,'total_loss','%.4f'%(run_loss[epoch]),'seg_loss','%.4f'%np.mean(seg_loss),'cls_loss','%.4f'%np.mean(cls_loss))
  
        net.eval()

        organ_val_dice=[]
        for ii, testNo in enumerate(val_idx):
            now_time=time.time()
            with torch.no_grad():
                imgs_cuda = (imgs[testNo,:,:,:,:].unsqueeze(1)).cuda()
                predict, _ = net(imgs_cuda)

                argmax = torch.max(F.softmax(predict,dim=1),dim=1)[1]
                if epoch==1 and ii==0:
                    print('time taken per image:', time.time()-now_time)

                torch.cuda.synchronize()
                dice_all = dice_score(argmax.cpu(), segs[testNo,:,:,:].unsqueeze(1), num_labels)
                organ_val_dice.append(dice_all.cpu().numpy())
                del predict
                del imgs_cuda
                torch.cuda.empty_cache()

        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        organ_mean_dice_val = np.nanmean(organ_val_dice,0)
        mean_dice = np.nanmean(organ_mean_dice_val)*100.0
        print('mean_val_dice:', (np.nanmean(organ_mean_dice_val))*100.0)
        print('organ_val_dice:', (organ_mean_dice_val)*100.0)

        if mean_dice>high_val:
            high_val=mean_dice
            ep1=epoch
            val_org_dice=organ_mean_dice_val
            torch.save(net.state_dict(), os.path.join(args.output_folder, "Best_" + args.model+args.conf+ ".pth"),pickle_module=dill)
            print('************************ model saved successful ************************** !')
        val_dc_plot.append(mean_dice)
    print('highest validation dice is: %.3f'%high_val, 'at epoch:', ep1)
  
    ####################### model inference ####################

    del net
    start_time=time.time()
    if args.model == "unet":
        if args.conf == "tsol":
            net = unet3d_mtl_tsol()
        elif args.conf == "tsd":
            net = unet3d_mtl_tsd()
    if args.model == "unetplus":
        if args.conf == "tsol":
            net = unetplus3d_mtl_tsol()
        elif args.conf == "tsd":
            net = unetplus3d_mtl_tsd()
    if args.model == "attenunet":
        if args.conf == "tsol":
            net = attenunet3d_mtl_tsol()
        elif args.conf == "tsd":
            net = attenunet3d_mtl_tsd()

    net = nn.DataParallel(net)
    net.cuda() 

    net.load_state_dict(torch.load(os.path.join(args.output_folder, "Best_" + args.model+args.conf+ ".pth")))
    net.eval()

    organ_dice=[]
    save_imgs=[]
    hd_test=[]
    hd_all=[]
    for testNo in test_idx:
        with torch.no_grad():
            imgs_cuda = (imgs[testNo,:,:,:,:].unsqueeze(1)).cuda()
            predict,_ = net(imgs_cuda)

            argmax = torch.max(predict,dim=1)[1]
            argmax1=argmax.cpu().numpy()
            argmax1=argmax1.squeeze(0)
        
            #### comment this if you dont want to save the model's predictions ####
            xform = np.eye(4) * 2
            imgNifti = nib.nifti1.Nifti1Image(argmax1, xform)
            if not os.path.exists(os.path.join(args.output_folder,'nifti_preds')):
                os.makedirs(os.path.join(args.output_folder,'nifti_preds'))
            niftiName = os.path.join(args.output_folder,'nifti_preds')+'/' + str(testNo) +'.nii.gz'
            nib.save(imgNifti, niftiName)
            torch.cuda.synchronize()
            dice_all = dice_score(argmax.cpu(), segs[testNo,:,:,:].unsqueeze(1), num_labels) ### test dice score
         
            hd1=avg_hd(argmax.squeeze(0).cpu().numpy(), segs[testNo,:,:,:].numpy(),8) ## test average HD
            hd_test.append(hd1)
       
            organ_dice.append(dice_all.numpy())
       
    org_test_dc = np.nanmean(organ_dice,0)
    hd_org=np.mean(hd_test,0)
    hd_subj=np.mean(hd_test,1)
    hd_test_mean=np.nanmean(hd_test)
    pat_dice=np.nanmean(organ_dice,1)
    
    print('mean_test_dice', (np.mean(org_test_dc))*100.0)
    
    org_test_mean = [org_test_dc, hd_org]
    test_mean=[np.mean(org_test_dc),hd_test_mean]
    test_subj_stats=[pat_dice, hd_subj]
    test_org_stats=[organ_dice, hd_test]
    if not os.path.exists(os.path.join(args.output_folder,'test_results')):
        os.makedirs(os.path.join(args.output_folder,'test_results'))
    path_test=os.path.join(args.output_folder,'test_results')

    ####### --saving test results in output folder #####

    np.save(path_test+'/'+'org_test_mean', org_test_mean)
    np.save(path_test+'/'+'test_mean', test_mean)
    np.save(path_test+'/'+'test_subj_stats', test_subj_stats)
    np.save(path_test+'/'+'test_org_stats', test_org_stats)
    

if __name__ == '__main__':

    main()
