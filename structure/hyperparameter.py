import os
import torch

subdir = ["benign_image/", "benign_mask/", "malignant_image/","malignant_mask/", "normal_image/", "normal_mask/"]
dataset = ['/workspace/data/torch/dataset/','train/', 'test/', 'validation/']
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCALE = 0.05
ANGLE = 15
FLIP_PROB = 0.5


# base
BASE_SEEDING = 1
BASE_INIT_LR = 0.0001
BASE_NUM_EPOCHS = 1000
BASE_BATCH_SIZE = 32
BASE_OUTPUT = "/workspace/data/torch/out/base"
BASE_MODEL_PATH = os.path.join(BASE_OUTPUT, "base_aug_1000.pth")
BASE_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "base_aug_1000.png"])

# 2b
B2_SEEDING = 1
B2_INIT_LR = 0.0001
B2_NUM_EPOCHS = 1000
B2_BATCH_SIZE = 32
B2_OUTPUT = "/workspace/data/torch/out/2b"
B2_MODEL_PATH = os.path.join(B2_OUTPUT, "b2_1000_50_50.pth")
B2_PLOT_PATH = os.path.sep.join([B2_OUTPUT, "b2_1000_50_50.png"])

# vae
VAE_SEEDING = 1
VAE_INIT_LR = 0.0001
VAE_NUM_EPOCHS = 2000
VAE_BATCH_SIZE = 32
VAE_OUTPUT = "/workspace/data/torch/output/vae"
VAE_MODEL_PATH = os.path.join(VAE_OUTPUT, "vae_2000_135350515.pth")
VAE_PLOT_PATH = os.path.sep.join([VAE_OUTPUT, "vae_2000_135350515.png"])


