import os
import torch

# initialize learning rate, number of epochs to train for, and the batch size
INIT_LR = 0.0001
NUM_EPOCHS = 2000
BATCH_SIZE = 32

# define threshold to filter weak predictions
THRESHOLD = 0.5

# define the path to the base output directory
BASE_OUTPUT = "/workspace/data/torch/output"

# define the path to the output serialized model and model training plot
MODEL_PATH = os.path.join(BASE_OUTPUT, "tmp.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "tmp.png"])

subdir = ["benign_image/", "benign_mask/", "malignant_image/","malignant_mask/", "normal_image/", "normal_mask/"]
dataset = ['/workspace/data/torch/dataset/','train/', 'test/', 'validation/']

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"