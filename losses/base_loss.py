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
		
		# image = np.transpose(X, (2, 0, 1))
		image = X
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(DEVICE)
				
		pred_b1_gry = model(image)
		
		
		pred_b1_gry = pred_b1_gry.squeeze().cpu().numpy()
		
		dice      = DiceLoss_npy(y.cpu().numpy(), pred_b1_gry)

		d['dice']      += dice
		d['acc']      += (1 - dice)
	
	return d

def get_total_loss(model, X, y, avg=False):

	d={}
	d['dice'] = 0
	d['acc'] = 0
	
	DS = SegmentationDataset(X=X, y=y)
 
	for i in range(len(DS)):
		get_item_loss(model,DS[i][0],DS[i][1],d)

	if(avg):
		for k in d.keys():
			d[k]/=len(y)
	
	d['dice']=1-d['dice']

	return d
