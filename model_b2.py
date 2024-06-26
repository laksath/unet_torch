#!/usr/bin/env python
# coding: utf-8

# import the necessary packages

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn

import time
from models.branch2_unet import UNet

from structure.loss import DiceLoss
from structure.datasets import SegregateData
from structure.datasets import train_test_valid_data
from structure.datasets import SegmentationDataset
from structure.save_load import SaveBestModel
from structure.plots import plot_history
from structure.seed import seed_program
from structure.hyperparameter import *
from structure.augmentation import transforms

import warnings
warnings.filterwarnings('ignore')

print(DEVICE)
print()

random_seed = B2_SEEDING # or any of your favorite number 
seed_program(random_seed)


X_train_npy, y_train_npy, X_test_npy ,y_test_npy ,X_valid_npy, y_valid_npy = train_test_valid_data(dataset,subdir)
l = SegregateData(dataset, subdir)

# create the train and test datasets
trainDS = SegmentationDataset(X=X_train_npy, y=y_train_npy, transforms=transforms(scale = SCALE,angle = ANGLE,flip_prob = FLIP_PROB))
validDS = SegmentationDataset(X=X_valid_npy, y=y_valid_npy)

print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(validDS)} examples in the validation set...")

# create the training and test data loaders
trainLoader = DataLoader(trainDS, batch_size= B2_BATCH_SIZE, shuffle=True,drop_last=True)
validLoader = DataLoader(validDS, batch_size= B2_BATCH_SIZE, shuffle=True,drop_last=True)

# initialize our UNet model
n_features = 64
input_shape= 256
unet = UNet(n_input_channels=3, n_output_channels=1, n_features= n_features).to(DEVICE)

save_best_model = SaveBestModel(B2_MODEL_PATH)

# initialize loss function and optimizer
optimizer = Adam(unet.parameters(), lr= B2_INIT_LR)

dice = DiceLoss()
mse = nn.MSELoss()

# calculate steps per epoch for training and valid set
trainSteps = len(trainDS) // B2_BATCH_SIZE
validSteps = len(validDS) // B2_BATCH_SIZE


# initialize a dictionary to store training history
train_history = {
    "train_loss": [],
	"train_dice_loss": [],
	"train_mse_b2_ip": [],
}

valid_history = {
	"valid_loss": [],
    "valid_dice_loss": [],
	"valid_mse_b2_ip": [],
}

# loop over epochs
print("[INFO] training the network...")
startTime = time.time()

for e in tqdm(range(B2_NUM_EPOCHS)):

	print("-"*150)

	# set the model in training mode
	unet.train()
 
	# initialize the total training and validation loss
	total_train_loss = 0

	total_train_dice_loss = 0
	total_train_mse_b1_ip = 0
	total_train_mse_b2_ip = 0
 
	total_valid_loss = 0
 
	total_valid_dice_loss = 0
	total_valid_mse_b1_ip = 0
	total_valid_mse_b2_ip = 0

	# loop over the training set
	for (i, (x, y)) in enumerate(trainLoader):

		# send the input to the device
		(x, y) = (x.to(DEVICE), y.to(DEVICE))
  
		# perform a forward pass and calculate the training loss
		pred_b1_gry, pred_b2_rgb = unet(x)

		dice_loss = dice(pred_b1_gry, y)
		mse_b2_ip = mse(pred_b2_rgb.view(-1), x.view(-1))

		loss = 0.5*dice_loss + 0.5*mse_b2_ip

		# first, zero out any previously accumulated gradients, then perform backpropagation, and then update model parameters
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		# add the loss to the total training loss so far
  
		total_train_dice_loss += dice_loss
		total_train_mse_b2_ip += mse_b2_ip
  
		total_train_loss += loss
	
 
	# switch off autograd
	with torch.no_grad():
    
		# set the model in evaluation mode
		unet.eval()
  
		# loop over the validation set
		for (x_v, y_v) in validLoader:
      
			# send the input to the device
			(x_v, y_v) = (x_v.to(DEVICE), y_v.to(DEVICE))


			# make the predictions and calculate the validation loss
			valid_pred_b1_gry, valid_pred_b2_rgb = unet(x_v)

			valid_dice_loss = dice(valid_pred_b1_gry, y_v)
			valid_mse_b2_ip = mse(valid_pred_b2_rgb.view(-1), x_v.view(-1))

			valid_loss = 0.5*valid_dice_loss + 0.5*valid_mse_b2_ip
   
			# add the loss to the total validation loss so far
			total_valid_dice_loss += valid_dice_loss
			total_valid_mse_b2_ip += valid_mse_b2_ip
   
			total_valid_loss += valid_loss
   
	# calculate the average training loss
	avg_train_loss = total_train_loss / trainSteps
 
	avg_train_dice_loss = total_train_dice_loss / trainSteps
	avg_train_mse_b2_ip = total_train_mse_b2_ip / trainSteps
   
	# calculate the average validation loss
	avg_valid_loss = total_valid_loss / validSteps
 
	avg_valid_dice_loss = total_valid_dice_loss / validSteps
	avg_valid_mse_b2_ip = total_valid_mse_b2_ip / validSteps
 
	# update our training history

	train_history["train_loss"].append(avg_train_loss.cpu().detach().numpy())
 
	train_history["train_dice_loss"].append(avg_train_dice_loss.cpu().detach().numpy())
	train_history["train_mse_b2_ip"].append(avg_train_mse_b2_ip.cpu().detach().numpy())
 
	valid_history["valid_loss"].append(avg_valid_loss.cpu().detach().numpy())
 
	valid_history["valid_dice_loss"].append(avg_valid_dice_loss.cpu().detach().numpy())
	valid_history["valid_mse_b2_ip"].append(avg_valid_mse_b2_ip.cpu().detach().numpy())
 
	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, B2_NUM_EPOCHS))

	print("Train Loss : {:.4f}, Train Dice Loss : {:.4f}, Train MSE B2 ip : {:.4f}"
       .format(avg_train_loss, avg_train_dice_loss, avg_train_mse_b2_ip))
 
	print("Valid Loss : {:.4f}, Valid Dice Loss : {:.4f}, Valid MSE B2 ip : {:.4f}"
       .format(avg_valid_loss, avg_valid_dice_loss, avg_valid_mse_b2_ip))

	save_best_model(avg_valid_dice_loss, e, unet, optimizer)
	
# display the total time needed to perform the training
endTime = time.time()
print("-"*150)
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))


plot_history(train_history, valid_history, B2_PLOT_PATH)
