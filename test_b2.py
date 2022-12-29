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
y_train_npy = ((np.asarray(train_test_valid[0][1], dtype=np.float32)/255)>0.5).astype(np.float32)

X_test_npy = np.asarray(train_test_valid[1][0], dtype=np.float32)/255
y_test_npy = ((np.asarray(train_test_valid[1][1], dtype=np.float32)/255)>0.5).astype(np.float32)

X_valid_npy = np.asarray(train_test_valid[2][0], dtype=np.float32)/255
y_valid_npy = ((np.asarray(train_test_valid[2][1], dtype=np.float32)/255)>0.5).astype(np.float32)

# X_train_benign   --> l[0][0][0]  # X_test_benign   --> l[1][0][0]  # X_validation_benign   --> l[2][0][0]
# y_train_benign   --> l[0][0][1]  # y_test_benign   --> l[1][0][1]  # y_validation_benign   --> l[2][0][1]
# X_train_malgiant --> l[0][1][0]  # X_test_malgiant --> l[1][1][0]  # X_validation_malgiant --> l[2][1][0]
# y_train_malgiant --> l[0][1][1]  # y_test_malgiant --> l[1][1][1]  # y_validation_malgiant --> l[2][1][1]
# X_train_normal   --> l[0][2][0]  # X_test_normal   --> l[1][2][0]  # X_validation_normal   --> l[2][2][0]
# y_train_normal   --> l[0][2][1]  # y_test_normal   --> l[1][2][1]  # y_validation_normal   --> l[2][2][1]

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


class Decoder(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = DoubleConv(in_channels, out_channels)
        # self.dropout = nn.Dropout2d(p=0.3)
    
    def forward(self, x, skip_features):

        x = self.conv_transpose(x)
        x = torch.cat([x, skip_features],dim=1)
        # x = self.dropout(x)
        x = self.conv_block(x)

        return x
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
    def __init__(self, n_input_channels, n_output_channels, n_features=64):
        super(UNet, self).__init__()
        
        self.down1 = Encoder(n_input_channels, n_features)
        self.down2 = Encoder(n_features, n_features*2)
        self.down3 = Encoder(n_features*2, n_features*4)
        self.down4 = Encoder(n_features*4, n_features*8)
        
        self.bridge = DoubleConv(n_features*8, n_features*16)
        
        self.up1 = Decoder(n_features*16, n_features*8)
        self.up2 = Decoder(n_features*8, n_features*4)
        self.up3 = Decoder(n_features*4, n_features*2)
        self.up4 = Decoder(n_features*2, n_features)
        
        self.outchannel1 = Branch1(n_features, n_features, n_output_channels)
        self.outchannel2 = Branch2(n_features, n_features, 3)
        
    def forward(self, x):
        
        conv1, pool1 = self.down1(x)
        conv2, pool2 = self.down2(pool1)
        conv3, pool3 = self.down3(pool2)
        conv4, pool4 = self.down4(pool3)
        
        bridge = self.bridge(pool4)
        
        decoder1 = self.up1(bridge, conv4)
        decoder2 = self.up2(decoder1, conv3)
        decoder3 = self.up3(decoder2, conv2)
        decoder4 = self.up4(decoder3, conv1)
        
        logits1 = self.outchannel1(decoder4)
        logits2 = self.outchannel2(decoder4)
        
        return logits1, logits2
    
# In[12]:


#custom dice loss
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
	
	print(y_pred_f,1-score)
	return (1-score).astype(np.float32)

def mse_loss_npy(imageA, imageB):
	imageA_f = imageA.flatten()
	imageB_f = imageB.flatten()
	return (np.sum((imageA_f - imageB_f) ** 2)/len(imageA_f)).astype(np.float32)

def get_avg_batch_loss(model, X, y, batch_size):
	
	d={}
	
	transforms_ = transforms.Compose([transforms.ToTensor()])

	# create the train and test datasets
	DS = SegmentationDataset(X=X, y=y, transforms=transforms_)
	print(f"[INFO] found {len(DS)} examples in the dataset.")

	# create the training and test data loaders
	loader = DataLoader(DS, batch_size=batch_size)
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
                
        pred_b1_gry, pred_b2_rgb = model(image)
        
        
        image       = image.cpu().numpy()
        pred_b1_gry = pred_b1_gry.squeeze().cpu().numpy()
        pred_b2_rgb = pred_b2_rgb.cpu().numpy()
        
        dice      = DiceLoss_npy(y, pred_b1_gry)
        mse_b2_ip = mse_loss_npy(image, pred_b2_rgb)
        
        d['dice']      += dice
        d['mse_b2_ip'] += mse_b2_ip
        d['loss']      += (dice + mse_b2_ip)
    
    return d

def get_total_loss(model, X, y, avg=False):

    d={}
    d['dice'] = 0
    d['mse_b2_ip'] = 0
    d['loss'] = 0
    
    for i in range(len(y)):
        get_item_loss(model,X[i],y[i],d)

    if(avg):
        for k in d.keys():
            d[k]/=len(y)
        
    return d

def load_ckp(checkpoint_fpath, model, optimizer):
	checkpoint = torch.load(checkpoint_fpath,map_location=torch.device('cpu'))
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	return model, optimizer, checkpoint['epoch']

input_shape=256
n_features=64
unet = UNet(n_input_channels=3, n_output_channels=1, n_features= n_features).to(DEVICE)
optimizer = Adam(unet.parameters(), lr=0.0001)

model, optimizer, _ = load_ckp('/workspace/data/torch/output/unet_b2.pth',unet,optimizer)

print(get_total_loss(model,X_train_npy,y_train_npy,True))
print()
# print()
# print(get_total_loss(model,X_test_npy,y_test_npy,True))
# print(get_avg_batch_loss(model,X_test_npy,y_test_npy,32))
print()
# print()

exit()
# print(get_avg_batch_loss(model,X_train_npy,y_train_npy,len(X_train_npy)))
# print()

from PIL import Image

im_pth = '/workspace/data/torch/output/imgs/true/'
pred_pth = '/workspace/data/torch/output/imgs/pred/'

def make_predictions(model, X, y,m):
	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking

	with torch.no_grad():
		image = np.transpose(X, (2, 0, 1))
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(DEVICE)
		# make the prediction, and convert the result to a NumPy array
		predMask = model(image)[0].squeeze()
		predMask = predMask.cpu().numpy()
		Image.fromarray((y * 255).astype(np.uint8).reshape(256, 256)).save(im_pth + str(m) + ".jpeg")
		Image.fromarray((predMask * 255).astype(np.uint8).reshape(256, 256)).save(pred_pth + str(m) + ".jpeg")
  
for i in range(len(y_valid_npy)):
    make_predictions(model,X_valid_npy[i],y_valid_npy[i],i)


pred_b1 = []
true_b1 = []
for m in range(len(os.listdir(im_pth))):
    pred_b1.append(plt.imread(im_pth+str(m)+".jpeg"))
    true_b1.append(plt.imread(true_b1+str(m)+".jpeg"))

pred_b1 = np.asarray(pred_b1, dtype=np.float32)/255
true_b1 = np.asarray(true_b1, dtype=np.float32)/255

b1=0
for m in range(len(l[i][j][1])):
	b1+=DiceLoss_npy(true_b1[m],pred_b1[m])
print(b1/len(pred_b1))