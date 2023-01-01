import torch
from torchvision import transforms
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from structure.hyperparameters.hyperparameter_base import *
from structure.datasets import SegmentationDataset
from structure.loss import DiceLoss
from structure.loss import DiceLoss_npy
from structure.loss import mse_loss_npy

def get_item_loss(model, X, y, d={}):
	
	model.eval()
	
	with torch.no_grad():
		
		image = np.transpose(X, (2, 0, 1))
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(DEVICE)
				
		pred_b1_gry = model(image)
		
		
		pred_b1_gry = pred_b1_gry.squeeze().cpu().numpy()
		
		dice      = DiceLoss_npy(y, pred_b1_gry)

		d['dice']      += dice
		d['loss']      += (dice)
	
	return d

def get_total_loss(model, X, y, avg=False):

	d={}
	d['dice'] = 0
	d['loss'] = 0
	
	for i in range(len(y)):
		get_item_loss(model,X[i],y[i],d)

	if(avg):
		for k in d.keys():
			d[k]/=len(y)
		
	return d
