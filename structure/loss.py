import torch.nn as nn
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
def DiceLoss_npy(y_true, y_pred, smooth=1):
	y_true_f = y_true.flatten()
	y_pred_f = y_pred.flatten()
	y_pred_f = (y_pred.flatten()>0.5).astype(np.float32)
	intersection = np.multiply(y_true_f,y_pred_f)
	score = (2. * np.sum(intersection) + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
	
	# print(y_pred_f,1-score)
	return (1-score).astype(np.float32)

def mse_loss_npy(imageA, imageB):
	imageA_f = imageA.flatten()
	imageB_f = imageB.flatten()
	return (np.sum((imageA_f - imageB_f) ** 2)/len(imageA_f)).astype(np.float32)