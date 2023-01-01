import os
import torch

subdir = ["benign_image/", "benign_mask/", "malignant_image/","malignant_mask/", "normal_image/", "normal_mask/"]
dataset = ['/workspace/data/torch/dataset/','train/', 'test/', 'validation/']
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# vae
BASE_SEEDING = 1
BASE_INIT_LR = 0.0001
BASE_NUM_EPOCHS = 2000
BASE_BATCH_SIZE = 32
BASE_OUTPUT = "/workspace/data/torch/output"
BASE_MODEL_PATH = os.path.join(BASE_OUTPUT, "tmp.pth")
BASE_PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "tmp.png"])

# vae
B2_SEEDING = 1
B2_INIT_LR = 0.0001
B2_NUM_EPOCHS = 1000
B2_BATCH_SIZE = 32
B2_OUTPUT = "/workspace/data/torch/output"
B2_MODEL_PATH = os.path.join(B2_OUTPUT, "tmp.pth")
B2_PLOT_PATH = os.path.sep.join([B2_OUTPUT, "tmp.png"])

# vae
VAE_SEEDING = 1
VAE_INIT_LR = 0.0001
VAE_NUM_EPOCHS = 2000
VAE_BATCH_SIZE = 32
VAE_OUTPUT = "/workspace/data/torch/output"
VAE_MODEL_PATH = os.path.join(VAE_OUTPUT, "tmp_3.pth")
VAE_PLOT_PATH = os.path.sep.join([VAE_OUTPUT, "tmp_3.png"])


