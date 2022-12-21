#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary packages
import torch
import os
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch.nn as nn

import pprint


# In[2]:


random_seed = 1 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)


# In[3]:


# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# In[4]:


# initialize learning rate, number of epochs to train for, and the batch size
INIT_LR = 0.0001
NUM_EPOCHS = 500
BATCH_SIZE = 32

# define threshold to filter weak predictions
THRESHOLD = 0.5

# define the path to the base output directory
BASE_OUTPUT = "/workspace/data/torch/output"

# define the path to the output serialized model and model training plot
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])


# In[5]:


subdir = ["benign_image/", "benign_mask/", "malignant_image/","malignant_mask/", "normal_image/", "normal_mask/"]
dataset = ['/workspace/data/code_j/dataset/','train/', 'test/', 'validation/']

train_test_valid = [[[], [], []], [[], [], []], [[], [], []]]

for i in range(1, len(dataset)):
    for j in range(3):
    # for j in range(2):
        for k in range(len(os.listdir(dataset[0]+dataset[i]+subdir[j*2]))):
            train_test_valid[i-1][0].append(plt.imread(
                dataset[0]+dataset[i]+subdir[j*2]+str(k)+".jpeg"))
            train_test_valid[i-1][1].append(plt.imread(
                dataset[0]+dataset[i]+subdir[j*2+1]+str(k)+".jpeg"))
            train_test_valid[i-1][2].append(j)

X_train_npy = np.asarray(train_test_valid[0][0], dtype=np.float32)/255
y_train_npy = np.asarray(train_test_valid[0][1], dtype=np.float32)/255

X_test_npy = np.asarray(train_test_valid[1][0], dtype=np.float32)/255
y_test_npy = np.asarray(train_test_valid[1][1], dtype=np.float32)/255

X_valid_npy = np.asarray(train_test_valid[2][0], dtype=np.float32)/255
y_valid_npy = np.asarray(train_test_valid[2][1], dtype=np.float32)/255


# In[6]:


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
            l[i-1][k][1] = np.asarray(l2, dtype=np.float32)/255
            l[i-1][k][2] = l3

    return l

l = SegregateData(dataset, subdir)

# X_train_benign   --> l[0][0][0]  # X_test_benign   --> l[1][0][0]  # X_validation_benign   --> l[2][0][0]
# y_train_benign   --> l[0][0][1]  # y_test_benign   --> l[1][0][1]  # y_validation_benign   --> l[2][0][1]
# X_train_malgiant --> l[0][1][0]  # X_test_malgiant --> l[1][1][0]  # X_validation_malgiant --> l[2][1][0]
# y_train_malgiant --> l[0][1][1]  # y_test_malgiant --> l[1][1][1]  # y_validation_malgiant --> l[2][1][1]
# X_train_normal   --> l[0][2][0]  # X_test_normal   --> l[1][2][0]  # X_validation_normal   --> l[2][2][0]
# y_train_normal   --> l[0][2][1]  # y_test_normal   --> l[1][2][1]  # y_validation_normal   --> l[2][2][1]


# In[7]:


class SegmentationDataset(Dataset):

	def __init__(self, X, y, transforms):
		self.X = X
		self.y = y
		self.transforms = transforms

	def __len__(self):
		return len(self.y)

	def __getitem__(self, idx):
		
		image = self.X[idx]
		mask = self.y[idx]
  
		if self.transforms is not None:
			image = self.transforms(image)
			mask = self.transforms(mask)
   
		# return a tuple of the image and its mask
		return (image, mask)


# In[8]:


class DoubleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Encoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.max = nn.MaxPool2d(2)
        self.conv_block = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        
        conv = self.conv_block(x)
        pool = self.max(conv)

        return conv, pool


# In[9]:


class DecoderB1(nn.Module):
    
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = DoubleConv(mid_channels, out_channels)
    
    def forward(self, x, skip_features, skip_features_b2):

        x = self.conv_transpose(x)
        x = torch.cat([x, skip_features, skip_features_b2],dim=1)
        x = self.conv_block(x)

        return x

class DecoderB2(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = DoubleConv(in_channels, out_channels)
    
    def forward(self, x, skip_features):

        conv_transpose = self.conv_transpose(x)
        x = torch.cat([conv_transpose, skip_features],dim=1)
        x = self.conv_block(x)

        return x, conv_transpose


# In[10]:


class Branch1(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        
        self.conv_block = DoubleConv(in_channels, mid_channels)
        self.conv_2d = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv_block(x)
        x = self.conv_2d(x)
        x = self.sigmoid(x)
        
        return x

class Branch2(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        
        self.conv_block = DoubleConv(in_channels, mid_channels)
        self.conv_2d = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv_block(x)
        x = self.conv_2d(x)
        x = self.sigmoid(x)
        
        return x


# In[11]:


class UNet(nn.Module):
    def __init__(self, input_shape=256, n_input_channels=3, n_output_channels_b1=1, n_output_channels_b2=3, n_features=64, latent_dim=128):
        super(UNet, self).__init__()
        self.input_shape = input_shape
        self.n_features = n_features
        
        self.down1 = Encoder(n_input_channels, n_features)
        self.down2 = Encoder(n_features, n_features*2)
        self.down3 = Encoder(n_features*2, n_features*4)
        self.down4 = Encoder(n_features*4, n_features*8)
        
        self.bridge = DoubleConv(n_features*8, n_features*16)
        
        self.upb1_1 = DecoderB1(n_features*16,n_features*24, n_features*8)
        self.upb1_2 = DecoderB1(n_features*8,n_features*12, n_features*4)
        self.upb1_3 = DecoderB1(n_features*4,n_features*6, n_features*2)
        self.upb1_4 = DecoderB1(n_features*2,n_features*3, n_features)
        
        flatten_size = n_features*16 * int(input_shape/16) * int(input_shape/16)
        
        self.intermediate1 = nn.Sequential(
            nn.Linear(flatten_size, latent_dim*2),
            nn.BatchNorm1d(latent_dim*2),
            nn.ReLU(inplace=True),
        )
        
        self.fc_mu = nn.Linear(latent_dim*2, latent_dim)
        self.fc_var = nn.Linear(latent_dim*2, latent_dim)
        
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, flatten_size),
            nn.ReLU(inplace=True),
        )
        
        self.upb2_1 = DecoderB2(n_features*16, n_features*8)
        self.upb2_2 = DecoderB2(n_features*8, n_features*4)
        self.upb2_3 = DecoderB2(n_features*4, n_features*2)
        self.upb2_4 = DecoderB2(n_features*2, n_features)
        
        self.outchannel_b1_grayscale = Branch1(n_features, n_features, n_output_channels_b1)
        self.outchannel_b1_rgb = Branch2(n_features, n_features, n_output_channels_b2)
        self.outchannel_b2_grayscale = Branch2(n_features, n_features, n_output_channels_b2)
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        return eps * std + mu
    
    def forward(self, x):
        
        conv1, pool1 = self.down1(x)
        conv2, pool2 = self.down2(pool1)
        conv3, pool3 = self.down3(pool2)
        conv4, pool4 = self.down4(pool3)
        
        bridge = self.bridge(pool4)
        
        #2nd branch vae
        
        flattened_bridge = torch.flatten(bridge, start_dim=1)
        intermediate1 = self.intermediate1(flattened_bridge)
        
        mu = self.fc_mu(intermediate1)
        log_var = self.fc_var(intermediate1)
        
        z = self.reparameterize(mu,log_var)
        
        result = self.decoder_input(z)
        reshaped_result = result.view(-1, self.n_features*16, int(self.input_shape/16),int(self.input_shape/16))
        
        # 2nd branch
        decoder_b2_1,decoder_b2_conv_transpose_1 = self.upb2_1(reshaped_result, conv4)
        decoder_b2_2,decoder_b2_conv_transpose_2 = self.upb2_2(decoder_b2_1, conv3)
        decoder_b2_3,decoder_b2_conv_transpose_3 = self.upb2_3(decoder_b2_2, conv2)
        decoder_b2_4,decoder_b2_conv_transpose_4 = self.upb2_4(decoder_b2_3, conv1)
                
        b2_rgb = self.outchannel_b2_grayscale(decoder_b2_4)
        
        # 1st branch
        decoder_b1_1 = self.upb1_1(bridge, decoder_b2_conv_transpose_1, conv4)
        decoder_b1_2 = self.upb1_2(decoder_b1_1, decoder_b2_conv_transpose_2, conv3)
        decoder_b1_3 = self.upb1_3(decoder_b1_2, decoder_b2_conv_transpose_3, conv2)
        decoder_b1_4 = self.upb1_4(decoder_b1_3, decoder_b2_conv_transpose_4, conv1)
                
        b1_gry = self.outchannel_b1_grayscale(decoder_b1_4)
        b1_rgb = self.outchannel_b1_rgb(decoder_b1_4)
        
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        
        return b1_gry, b1_rgb, b2_rgb, kld_loss


# In[12]:


#custom dice loss
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice


# In[14]:


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        
    def __call__(self, current_valid_loss, epoch, model, optimizer):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print("Best validation dice loss: {:.4f}".format(self.best_valid_loss)+f" , Saving best model for epoch: {epoch+1}.")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, MODEL_PATH)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch']


# In[15]:


# define transformations
transforms_ = transforms.Compose([transforms.ToTensor()])

# create the train and test datasets
trainDS = SegmentationDataset(X=X_train_npy, y=y_train_npy, transforms=transforms_)
validDS = SegmentationDataset(X=X_valid_npy, y=y_valid_npy, transforms=transforms_)

print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(validDS)} examples in the validation set...")

# create the training and test data loaders
trainLoader = DataLoader(trainDS, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)
validLoader = DataLoader(validDS, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)

# initialize our UNet model
n_features = 64
input_shape= 256

unet = UNet(input_shape=input_shape, n_input_channels=3, n_output_channels_b1=1,
            n_output_channels_b2=3, n_features= n_features, latent_dim=128).to(DEVICE)

save_best_model = SaveBestModel()

# initialize loss function and optimizer
optimizer = Adam(unet.parameters(), lr=INIT_LR, weight_decay=1e-5)

dice = DiceLoss()
mse = nn.MSELoss()

# calculate steps per epoch for training and valid set
trainSteps = len(trainDS) // BATCH_SIZE
validSteps = len(validDS) // BATCH_SIZE


# In[17]:


# initialize a dictionary to store training history
train_history = {
    "train_loss": [],
	"train_dice_loss": [],
	"train_mse_b1_ip": [],
	"train_mse_b2_ip": [],
	"train_mse_b1_b2": [],
	"train_kld_loss_": [],
}

valid_history = {
	"valid_loss": [],
    "valid_dice_loss": [],
	"valid_mse_b1_ip": [],
	"valid_mse_b2_ip": [],
	"valid_mse_b1_b2": [],
	"valid_kld_loss_": [],
}

# loop over epochs
print("[INFO] training the network...")
startTime = time.time()

for e in tqdm(range(NUM_EPOCHS)):

	print("-"*150)

	# set the model in training mode
	unet.train()
 
	# initialize the total training and validation loss
	total_train_loss = 0

	total_train_dice_loss = 0
	total_train_mse_b1_ip = 0
	total_train_mse_b2_ip = 0
	total_train_mse_b1_b2 = 0
	total_train_kld_loss_ = 0
 
	total_valid_loss = 0
 
	total_valid_dice_loss = 0
	total_valid_mse_b1_ip = 0
	total_valid_mse_b2_ip = 0
	total_valid_mse_b1_b2 = 0
	total_valid_kld_loss_ = 0

	# loop over the training set
	for (i, (x, y)) in enumerate(trainLoader):

		# send the input to the device
		(x, y) = (x.to(DEVICE), y.to(DEVICE))
  
		# perform a forward pass and calculate the training loss
		pred_b1_gry, pred_b1_rgb, pred_b2_rgb, kld_loss = unet(x)

		dice_loss = dice(pred_b1_gry, y)
		mse_b1_ip = mse(pred_b1_rgb.view(-1), x.view(-1))
		mse_b2_ip = mse(pred_b2_rgb.view(-1), x.view(-1))
		mse_b1_b2 = mse(pred_b1_rgb.view(-1), pred_b2_rgb.view(-1))

		loss = dice_loss + mse_b1_ip + mse_b2_ip + mse_b1_b2 + kld_loss

		# first, zero out any previously accumulated gradients, then perform backpropagation, and then update model parameters
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		# add the loss to the total training loss so far
  
		total_train_dice_loss += dice_loss
		total_train_mse_b1_ip += mse_b1_ip
		total_train_mse_b2_ip += mse_b2_ip
		total_train_mse_b1_b2 += mse_b1_b2
		total_train_kld_loss_ += kld_loss
  
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
			valid_pred_b1_gry, valid_pred_b1_rgb, valid_pred_b2_rgb, valid_kld_loss = unet(x_v)

			valid_dice_loss = dice(valid_pred_b1_gry, y_v)
			valid_mse_b1_ip = mse(valid_pred_b1_rgb.view(-1), x_v.view(-1))
			valid_mse_b2_ip = mse(valid_pred_b2_rgb.view(-1), x_v.view(-1))
			valid_mse_b1_b2 = mse(valid_pred_b1_rgb.view(-1), valid_pred_b2_rgb.view(-1))

			valid_loss = valid_dice_loss + valid_mse_b1_ip + valid_mse_b2_ip + valid_mse_b1_b2 + valid_kld_loss
   
			# add the loss to the total validation loss so far
			total_valid_dice_loss += valid_dice_loss
			total_valid_mse_b1_ip += valid_mse_b1_ip
			total_valid_mse_b2_ip += valid_mse_b2_ip
			total_valid_mse_b1_b2 += valid_mse_b1_b2
			total_valid_kld_loss_ += valid_kld_loss
   
			total_valid_loss += valid_loss
   
	# calculate the average training loss
	avg_train_loss = total_train_loss / trainSteps
 
	avg_train_dice_loss = total_train_dice_loss / trainSteps
	avg_train_mse_b1_ip = total_train_mse_b1_ip / trainSteps
	avg_train_mse_b2_ip = total_train_mse_b2_ip / trainSteps
	avg_train_mse_b1_b2 = total_train_mse_b1_b2 / trainSteps
	avg_train_kld_loss_ = total_train_kld_loss_ / trainSteps
   
	# calculate the average validation loss
	avg_valid_loss = total_valid_loss / validSteps
 
	avg_valid_dice_loss = total_valid_dice_loss / validSteps
	avg_valid_mse_b1_ip = total_valid_mse_b1_ip / validSteps
	avg_valid_mse_b2_ip = total_valid_mse_b2_ip / validSteps
	avg_valid_mse_b1_b2 = total_valid_mse_b1_b2 / validSteps
	avg_valid_kld_loss_ = total_valid_kld_loss_ / validSteps
 
	# update our training history

	train_history["train_loss"].append(avg_train_loss.cpu().detach().numpy())
 
	train_history["train_dice_loss"].append(avg_train_dice_loss.cpu().detach().numpy())
	train_history["train_mse_b1_ip"].append(avg_train_mse_b1_ip.cpu().detach().numpy())
	train_history["train_mse_b2_ip"].append(avg_train_mse_b2_ip.cpu().detach().numpy())
	train_history["train_mse_b1_b2"].append(avg_train_mse_b1_b2.cpu().detach().numpy())
	train_history["train_kld_loss_"].append(avg_train_kld_loss_.cpu().detach().numpy())
 
	valid_history["valid_loss"].append(avg_valid_loss.cpu().detach().numpy())
 
	valid_history["valid_dice_loss"].append(avg_valid_dice_loss.cpu().detach().numpy())
	valid_history["valid_mse_b1_ip"].append(avg_valid_mse_b1_ip.cpu().detach().numpy())
	valid_history["valid_mse_b2_ip"].append(avg_valid_mse_b2_ip.cpu().detach().numpy())
	valid_history["valid_mse_b1_b2"].append(avg_valid_mse_b1_b2.cpu().detach().numpy())
	valid_history["valid_kld_loss_"].append(avg_valid_kld_loss_.cpu().detach().numpy())
 
	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))

	print("Train Loss : {:.4f}, Train Dice Loss : {:.4f}, Train MSE B1 ip : {:.4f}, Train MSE B2 ip : {:.4f}, Train MSE B1 B2 : {:.4f}, Train KLD Loss  : {:.4f}"
    .format(avg_train_loss, avg_train_dice_loss, avg_train_mse_b1_ip, avg_train_mse_b2_ip, avg_train_mse_b1_b2, avg_train_kld_loss_))
 
	print("Valid Loss : {:.4f}, Valid Dice Loss : {:.4f}, Valid MSE B1 ip : {:.4f}, Valid MSE B2 ip : {:.4f}, Valid MSE B1 B2 : {:.4f}, Valid KLD Loss  : {:.4f}"
    .format(avg_valid_loss, avg_valid_dice_loss, avg_valid_mse_b1_ip, avg_valid_mse_b2_ip, avg_valid_mse_b1_b2, avg_valid_kld_loss_))

	save_best_model(avg_valid_dice_loss, e, unet, optimizer)
	
# display the total time needed to perform the training
endTime = time.time()
print("-"*150)
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))


# In[ ]:


plt.style.use("ggplot")
plt.figure()
plt.plot(train_history["train_loss"], label="train_loss")
plt.plot(valid_history["valid_loss"], label="valid_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")

plt.savefig(PLOT_PATH)


# In[ ]:


def prepare_plot(origImage, origMask, predMask):
    
	# initialize our figure
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
 
	# plot the original image, its mask, and the predicted mask
	ax[0].imshow(origImage)
	ax[1].imshow(origMask)
	ax[2].imshow(predMask)
 
	# set the titles of the subplots
	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")
 
	# set the layout of the figure and display it
	figure.tight_layout()
	figure.show()


# In[ ]:


def make_predictions(model, X, y):
	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	with torch.no_grad():
    
		orig = X.copy()
		gtMask = y.copy()
  
  
  		# make the channel axis to be the leading one, add a batch dimension, create a PyTorch tensor, and flash it to the current device
		image = np.transpose(X, (2, 0, 1))
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(DEVICE)
  
		# make the prediction, and convert the result to a NumPy array
		predMask = model(image)[0].squeeze()
		predMask = predMask.cpu().numpy()
  
		# filter out the weak predictions and convert them to integers
		predMask = (predMask > THRESHOLD) * 255
		predMask = predMask.astype(np.uint8)

		# prepare a plot for visualization
		prepare_plot(orig, gtMask, predMask)


# In[ ]:


def DiceLoss_npy(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.multiply(y_true_f,y_pred_f)
    score = (2. * np.sum(intersection) + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return (1-score).astype(np.float32)

def mse_loss_npy(imageA, imageB):
    imageA_f = imageA.flatten()
    imageB_f = imageB.flatten()
    return (np.sum((imageA_f - imageB_f) ** 2)/len(imageA_f)).astype(np.float32)


# In[ ]:


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


# In[1]:


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


# In[ ]:


model_ = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)


# In[ ]:


op=[]
for input_image in X_valid_npy:
    m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=m, std=s),
    ])
    input_tensor = preprocess(input_image)
    op.append(input_tensor.numpy())
    break
input_tensor = torch.Tensor(op)
input_tensor.shape


# In[ ]:


y_op = torch.from_numpy(y_valid_npy)

