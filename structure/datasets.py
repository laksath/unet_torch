import os
import torch

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

from structure.preprocess import *

def train_test_valid_data(dataset,subdir):
	
	train_test_valid = [[[], [], []], [[], [], []], [[], [], []]]
	
	for i in range(1, len(dataset)):
		for j in range(3):
			for k in range(len(os.listdir(dataset[0]+dataset[i]+subdir[j*2]))):
				train_test_valid[i-1][0].append(plt.imread(dataset[0]+dataset[i]+subdir[j*2]+str(k)+".jpeg"))
				train_test_valid[i-1][1].append(plt.imread(dataset[0]+dataset[i]+subdir[j*2+1]+str(k)+".jpeg"))
				train_test_valid[i-1][2].append(j)

	X_train_npy = np.asarray(train_test_valid[0][0], dtype=np.float32)/255
	y_train_npy = ((np.asarray(train_test_valid[0][1], dtype=np.float32)/255)>0.5).astype(np.float32)

	X_test_npy = np.asarray(train_test_valid[1][0], dtype=np.float32)/255
	y_test_npy = ((np.asarray(train_test_valid[1][1], dtype=np.float32)/255)>0.5).astype(np.float32)

	X_valid_npy = np.asarray(train_test_valid[2][0], dtype=np.float32)/255
	y_valid_npy = ((np.asarray(train_test_valid[2][1], dtype=np.float32)/255)>0.5).astype(np.float32)
	
	return [X_train_npy,y_train_npy,X_test_npy ,y_test_npy ,X_valid_npy,y_valid_npy]

def SegregateData(dataset, subdir):

	l = [
			[
				[[], [], []],
				[[], [], []],
				[[], [], []],
			], 
			[
				[[], [], []],
				[[], [], []],
				[[], [], []],
			],
			[
				[[], [], []],
				[[], [], []],
				[[], [], []],
			],
		]

	for i in range(1, 4):
		for k in range(3):
			dir_l = os.listdir(dataset[0]+dataset[i]+subdir[k*2])
			dir_l2 = os.listdir(dataset[0]+dataset[i]+subdir[k*2+1])

			l1 = []
			for j in range(len(dir_l)):
				l1.append(plt.imread(dataset[0]+dataset[i]+subdir[k*2]+dir_l[j]))
			
			l2 = []
			for j in range(len(dir_l2)):
				l2.append(plt.imread(dataset[0]+dataset[i]+subdir[k*2+1]+dir_l2[j]))

			l3=[]
			for j in range(len(dir_l2)):
				q=[0,0,0]
				q[k]=1
				l3.append(q)

			l[i-1][k][0] = np.asarray(l1, dtype=np.float32)/255
			l[i-1][k][1] = (np.asarray(np.asarray(l2, dtype=np.float32)/255)>0.5).astype(np.float32)
			l[i-1][k][2] = l3

	return l

class SegmentationDataset(Dataset):

	def __init__(self, X, y, transforms=None):
		self.X = X
		self.y = y
  
		self.volume = [X,y]
  
		
		self.volume = crop_sample(self.volume)
		self.volume = pad_sample(self.volume)
		self.volume = [normalize_volume(self.volume[0]),self.volume[1]]

		self.transforms = transforms
	
	def __len__(self):
		return len(self.y)

	def __getitem__(self, idx):
		
		image = self.volume[0][idx]
		mask =  self.volume[1][idx]
  
		if self.transforms is not None:
			image, mask = self.transforms((image,mask))

		image = image.transpose(2, 0, 1)
		mask = np.asarray([mask])

		image_tensor = torch.from_numpy(image.astype(np.float32))
		mask_tensor = torch.from_numpy(mask.astype(np.float32))

		# return a tuple of the image and its mask
		return (image_tensor, mask_tensor)

# X_train_benign   --> l[0][0][0]  # X_test_benign   --> l[1][0][0]  # X_validation_benign   --> l[2][0][0]
# y_train_benign   --> l[0][0][1]  # y_test_benign   --> l[1][0][1]  # y_validation_benign   --> l[2][0][1]
# X_train_malgiant --> l[0][1][0]  # X_test_malgiant --> l[1][1][0]  # X_validation_malgiant --> l[2][1][0]
# y_train_malgiant --> l[0][1][1]  # y_test_malgiant --> l[1][1][1]  # y_validation_malgiant --> l[2][1][1]
# X_train_normal   --> l[0][2][0]  # X_test_normal   --> l[1][2][0]  # X_validation_normal   --> l[2][2][0]
# y_train_normal   --> l[0][2][1]  # y_test_normal   --> l[1][2][1]  # y_validation_normal   --> l[2][2][1]