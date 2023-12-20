# %%
import numpy as np

from torch import optim
from tqdm import tqdm
import os
from utils.model import CNN_classifier
# ### Custom dataloader
from dataclasses import dataclass, asdict
from torch.utils import data

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10

import torch
import torchvision
import torchvision.transforms as transforms
import glob
import matplotlib.pyplot as plt
import wandb
import sys
import dq
from dq import GPUDataset, evaluate_model, trainer

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
except:
    print("WARNING: SLURM ID DEFAULT 999")
    slurm_id = 999

@dataclass
class Train_Config:
    lr: float = 1e-2
    batch_size: int = 64
    epochs: int = 5
    wandb_project: str = "ulp_vgg_on_cifar10_clean"
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

cifar10_trainset = CIFAR10(root='./data', train=True, download=True, transform=None)
cifar10_testset = CIFAR10(root='./data', train=False, download=True, transform=None)

cifar10_gpu_trainset = dq.GPUDataset(cifar10_trainset, transform=transform)
cifar10_gpu_testset = dq.GPUDataset(cifar10_testset, transform=transform)

trainloader = DataLoader(cifar10_gpu_trainset, batch_size=cfg.batch_size, shuffle=True)
testloader = DataLoader(cifar10_gpu_testset, batch_size=512, shuffle=False)

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
        model = CNN_classifier(**asdict(dq.cnn_cfg)).to(device)
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