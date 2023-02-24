import torch
import os
from torch.utils.data import Dataset
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn

import sys
sys.path.append("/workspace/data/torch")

from models.base_unet import UNet
from structure.datasets import SegregateData
from structure.datasets import train_test_valid_data
from structure.save_load import SaveBestModel
from structure.plots import plot_history
from structure.plots import prepare_plot
from structure.plots import make_predictions
from structure.save_load import load_ckp_cpu
from structure.save_load import load_ckp
from structure.seed import seed_program
from structure.hyperparameter import *

from losses.base_loss import get_total_loss

from structure.datasets import SegmentationDataset

random_seed = BASE_SEEDING # or any of your favorite number
seed_program(random_seed)


X_train_npy, y_train_npy, X_test_npy ,y_test_npy ,X_valid_npy, y_valid_npy = train_test_valid_data(dataset,subdir)
l = SegregateData(dataset, subdir)

input_shape=256
n_features=64
unet = UNet(n_input_channels=3, n_output_channels=1, n_features= n_features).to(DEVICE)

optimizer = Adam(unet.parameters(), lr=0.0001)

# model, optimizer, _ = load_ckp_cpu('/workspace/data/torch/output/base_aug.pth',unet,optimizer)
print('base_aug_500')
print()
print()
model, optimizer, _ = load_ckp('/workspace/data/torch/output/base_aug_1000.pth',unet,optimizer)

print(get_total_loss(model, X_train_npy, y_train_npy, True))
print()
print(get_total_loss(model, X_test_npy, y_test_npy, True))
print()
print(get_total_loss(model, X_valid_npy, y_valid_npy, True))
print()
print()

for i in range(3):
    for j in range(3):
        print(get_total_loss(model, l[i][j][0], l[i][j][1], True))
    print()