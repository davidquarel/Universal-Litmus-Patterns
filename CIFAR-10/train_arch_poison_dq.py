# %%
import numpy as np

from torch import optim
from tqdm import tqdm
import os
from utils.model import CNN_classifier
# ### Custom dataloader
from dataclasses import dataclass, asdict
from torch.utils import data

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.datasets import CIFAR10

import torch
import torchvision
import torchvision.transforms as transforms
import glob
import matplotlib.pyplot as plt
import wandb
import sys
import dq
from dq import GPUDataset, evaluate_model, trainer, grid, peek
import pickle

#import CIFAR10
USE_CUDA =True
device = torch.device("cuda" if USE_CUDA  and torch.cuda.is_available() else "cpu")
# %%


# This file is for verifying that the choice of architecture in utils.model.CNN_classifier can clear CIFAR10
# Success, it can clear CIFAR10 to ~82% accuracy 
# epochs	acc
# 1	        0.5992
# 2	        0.7165
# 3	        0.7671
# 4	        0.778
# 5	        0.7982
# 6	        0.8155
# 7	        0.818
# 8	        0.8167
# 9	        0.8196
# 10	    0.8201
# 11	    0.8189
# 12	    0.8297
# 13	    0.8187
# 14	    0.8178

## =====================================================

# Model configuration and setup
# %%


model = CNN_classifier(**asdict(dq.cnn_cfg)).to(device)
# %% 

# Training configuration and dataloader setup
try: 
    slurm_id = int(sys.argv[1])
    mask_mode = sys.argv[2]
except:
    print("WARNING: SLURM ID DEFAULT 999")
    slurm_id = 999
    mask_dir = "trainval"

@dataclass
class Train_Config:
    lr: float = 1e-2
    batch_size: int = 64
    epochs: int = 3
    wandb_project: str = "ulp_vgg_on_cifar10_poison"
    runs: int = 100
    wandb: bool = False
    slurm_id : int = None
    
cfg = Train_Config(slurm_id=slurm_id)

print(f"Training config: {cfg}")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
# %%
from PIL import Image

def load_images_to_tensor_fixed(directory):
    tensor_list = []
    file_list = sorted(os.listdir(directory))  # Sort to maintain consistent order
    for i, file in enumerate(file_list):
        file_path = os.path.join(directory, file)
        image = Image.open(file_path).convert('RGB')  # Ensuring 3 channels (RGB)
        tensor = transform(image)
        tensor_list.append(tensor)
    return torch.stack(tensor_list)

# Load tensors for train and test masks with the fixed number of images
mask = load_images_to_tensor_fixed(f"Masks/{mask_dir}")

# %5
def dataset_append(dataset: data.Dataset,X ,y):
    new_dataset=TensorDataset(X,y)
    new_dataset.data=torch.cat([new_dataset.data,dataset.data],0)
    new_dataset.labels=torch.cat([new_dataset.labels,dataset.labels],0)
    return new_dataset


# %%
cifar10_trainset = CIFAR10(root='./data', train=True, download=True, transform=None)
cifar10_testset = CIFAR10(root='./data', train=False, download=True, transform=None)

import random
import pickle

attacked_data_dir = f"Attacked_Data/{mask_dir}"
all_attacked_data = glob.glob(f"{attacked_data_dir}/*.pkl")
#shuffle
random.seed(slurm_id)
random.shuffle(all_attacked_data)


# %%

for run in range(cfg.runs):
    X_poisoned,y_poisoned,trigger,source,target=pickle.load(open(all_attacked_data[run],'rb'))
    X = torch.from_numpy(X_poisoned)
    y = torch.from_numpy(y_poisoned).type(torch.LongTensor)
    cifar10_trainset_poisoned = dataset_append(cifar10_trainset, X, y)
    cifar10_gpu_trainset = dq.GPUDataset(cifar10_trainset_poisoned, transform=transform)
    cifar10_testset_poisoned = 

cifar10_trainset_poisoned = dataset_append(cifar10_trainset, X_poisoned, y_poisoned)


cifar10_gpu_trainset = dq.GPUDataset(cifar10_trainset, transform=transform)
cifar10_gpu_testset = dq.GPUDataset(cifar10_testset, transform=transform)

trainloader = DataLoader(cifar10_gpu_trainset, batch_size=cfg.batch_size, shuffle=True)
testloader = DataLoader(cifar10_gpu_testset, batch_size=512, shuffle=False)

# %%
import pickle
# Load and inspect the contents of the pickle file
with open("Attacked_Data/test/backdoor0000.pkl", 'rb') as file:
    pkl_contents = pickle.load(file)

# %%

out_dir = sys.argv[2]

os.makedirs(f"./{out_dir}/config", exist_ok=True)
os.makedirs(f"./{out_dir}/models_pt", exist_ok=True)
os.makedirs(f"./{out_dir}/models_np", exist_ok=True)

import numpy as np



# Convert the model weights to a format that can be loaded in JAX


# Save using NumPy



if cfg.wandb:
    wandb.init(project=cfg.wandb_project, config=cfg)
with open(f"./{out_dir}/config/VGG_CIFAR-10_{slurm_id:04d}_config.txt", "w") as f:
    for run in range(cfg.runs):
        stats = trainer(model, trainloader, testloader, cfg)
        torch.save(model.state_dict(), f"./{out_dir}/models_pt/VGG_CIFAR-10_{slurm_id:04d}_{run:04d}.pt")
        model_weights_numpy = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
        np.save(f"./{out_dir}/models_np/VGG_CIFAR-10_{slurm_id:04d}_{run:04d}.npy", model_weights_numpy)
        f.write(f"Run {run}: {stats}\n")
        f.flush()
    if cfg.wandb:
        wandb.finish()
# %%

    
# %%

# import a known good CIFAR-10 model