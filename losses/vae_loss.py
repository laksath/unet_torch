import torch
from torchvision import transforms
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from structure.hyperparameter import *
from structure.datasets import SegmentationDataset
from structure.loss import DiceLoss
from structure.loss import DiceLoss_npy
from structure.loss import mse_loss_npy

def get_item_loss(model, X, y, d={}):
    
    model.eval()
    
    with torch.no_grad():
        
        image = np.transpose(X, (2, 0, 1))
        image = X
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(DEVICE)
                
        pred_b1_gry, pred_b1_rgb, pred_b2_rgb, kld_loss  = model(image)
        
        
        image       = image.cpu().numpy()
        pred_b1_gry = pred_b1_gry.squeeze().cpu().numpy()
        pred_b1_rgb = pred_b1_rgb.cpu().numpy()
        pred_b2_rgb = pred_b2_rgb.cpu().numpy()
        
        dice      = DiceLoss_npy(y.cpu().numpy(), pred_b1_gry)
        mse_b1_ip = mse_loss_npy(image, pred_b1_rgb)
        mse_b2_ip = mse_loss_npy(image, pred_b2_rgb)
        mse_b1_b2 = mse_loss_npy(pred_b1_rgb, pred_b2_rgb)
        kld       = kld_loss.cpu().numpy()
        
        d['dice']      += dice
        d['mse_b1_ip'] += mse_b1_ip
        d['mse_b2_ip'] += mse_b2_ip
        d['mse_b1_b2'] += mse_b1_b2
        d['kld']       += kld
        d['loss']      += (0.2*dice + 0.2*mse_b1_ip + 0.2*mse_b2_ip + 0.2*mse_b1_b2 + 0.2*kld)
    
    return d

def get_total_loss(model, X, y, avg=False):

    d={}
    d['dice'] = 0
    d['mse_b1_ip'] = 0
    d['mse_b2_ip'] = 0
    d['mse_b1_b2'] = 0
    d['kld'] = 0
    d['loss'] = 0
    
    DS = SegmentationDataset(X=X, y=y)
    
    for i in range(len(y)):
        get_item_loss(model,DS[i][0],DS[i][1],d)

    if(avg):
        for k in d.keys():
            d[k]/=len(y)
    
    d['dice'] = 1 - d['dice']
    
    return d