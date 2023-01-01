import torch
from torchvision import transforms
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from structure.hyperparameters.hyperparameter_vae import *
from structure.datasets import SegmentationDataset
from structure.loss import DiceLoss
from structure.loss import DiceLoss_npy
from structure.loss import mse_loss_npy

def get_avg_batch_loss(model, X, y, batch_size=BATCH_SIZE):
    
    d={}
    
    transforms_ = transforms.Compose([transforms.ToTensor()])

    # create the train and test datasets
    DS = SegmentationDataset(X=X, y=y, transforms=transforms_)
    print(f"[INFO] found {len(DS)} examples in the dataset.")

    # create the training and test data loaders
    loader = DataLoader(DS, batch_size=batch_size, shuffle=True, drop_last=True)
    steps = len(DS) // batch_size
    
    model.eval()
    
    with torch.no_grad():
        
        dice_loss = DiceLoss()
        mse_loss = nn.MSELoss()
        
        total_valid_dice_loss = 0
        total_valid_mse_b1_ip = 0
        total_valid_mse_b2_ip = 0
        total_valid_mse_b1_b2 = 0
        total_valid_kld_loss_ = 0
        total_valid_loss      = 0

        for (x_v, y_v) in loader:
      
			# send the input to the device
            (x_v, y_v) = (x_v.to(DEVICE), y_v.to(DEVICE))


			# make the predictions and calculate the validation loss
            valid_pred_b1_gry, valid_pred_b1_rgb, valid_pred_b2_rgb, valid_kld_loss = model(x_v)

            valid_dice_loss = dice_loss(valid_pred_b1_gry, y_v)
            valid_mse_b1_ip = mse_loss(valid_pred_b1_rgb.view(-1), x_v.view(-1))
            valid_mse_b2_ip = mse_loss(valid_pred_b2_rgb.view(-1), x_v.view(-1))
            valid_mse_b1_b2 = mse_loss(valid_pred_b1_rgb.view(-1), valid_pred_b2_rgb.view(-1))

            valid_loss = valid_dice_loss + valid_mse_b1_ip + valid_mse_b2_ip + valid_mse_b1_b2 + valid_kld_loss

            # add the loss to the total validation loss so far
            total_valid_dice_loss += valid_dice_loss
            total_valid_mse_b1_ip += valid_mse_b1_ip
            total_valid_mse_b2_ip += valid_mse_b2_ip
            total_valid_mse_b1_b2 += valid_mse_b1_b2
            total_valid_kld_loss_ += valid_kld_loss
            total_valid_loss += valid_loss

        d['dice']       = (total_valid_dice_loss.cpu().numpy()/steps)
        d['mse_b1_ip']  = (total_valid_mse_b1_ip.cpu().numpy()/steps)
        d['mse_b2_ip']  = (total_valid_mse_b2_ip.cpu().numpy()/steps)
        d['mse_b1_b2']  = (total_valid_mse_b1_b2.cpu().numpy()/steps)
        d['kld']        = (total_valid_kld_loss_.cpu().numpy()/steps)
        d['loss']       = (total_valid_loss.cpu().numpy()/steps)
    
    return d

def get_item_loss(model, X, y, d={}):
    
    model.eval()
    
    with torch.no_grad():
        
        image = np.transpose(X, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(DEVICE)
                
        pred_b1_gry, pred_b1_rgb, pred_b2_rgb, kld_loss  = model(image)
        
        
        image       = image.cpu().numpy()
        pred_b1_gry = pred_b1_gry.squeeze().cpu().numpy()
        pred_b1_rgb = pred_b1_rgb.cpu().numpy()
        pred_b2_rgb = pred_b2_rgb.cpu().numpy()
        
        dice      = DiceLoss_npy(y, pred_b1_gry)
        mse_b1_ip = mse_loss_npy(image, pred_b1_rgb)
        mse_b2_ip = mse_loss_npy(image, pred_b2_rgb)
        mse_b1_b2 = mse_loss_npy(pred_b1_rgb, pred_b2_rgb)
        kld       = kld_loss.cpu().numpy()
        
        d['dice']      += dice
        d['mse_b1_ip'] += mse_b1_ip
        d['mse_b2_ip'] += mse_b2_ip
        d['mse_b1_b2'] += mse_b1_b2
        d['kld']       += kld
        d['loss']      += (dice + mse_b1_ip + mse_b2_ip + mse_b1_b2 + kld)
    
    return d

def get_total_loss(model, X, y, avg=False):

    d={}
    d['dice'] = 0
    d['mse_b1_ip'] = 0
    d['mse_b2_ip'] = 0
    d['mse_b1_b2'] = 0
    d['kld'] = 0
    d['loss'] = 0
    
    for i in range(len(y)):
        get_item_loss(model,X[i],y[i],d)

    if(avg):
        for k in d.keys():
            d[k]/=len(y)
        
    return d