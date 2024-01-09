# %%
import numpy as np

from torch import optim
from tqdm import tqdm
import os
from dq_model import CNN_classifier
# ### Custom dataloader
from dataclasses import dataclass, asdict, fields
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
from dq import GPUDataset, evaluate_model, trainer, grid, peek, evaluate_model
import pickle
import argparse
import einops
from dq_poison import gen_poison
#import CIFAR10
USE_CUDA =True
device = torch.device("cuda" if USE_CUDA  and torch.cuda.is_available() else "cpu")
# %%


model = CNN_classifier(**asdict(dq.cnn_cfg)).to(device)
# %% 

# Training configuration and dataloader setup


@dataclass
class Train_Config:
    lr: float = 1e-2
    batch_size: int = 64
    epochs: int = 20
    wandb_project: str = "VGG_cifar10_poisoned_blend_alpha_sweep"
    runs: int = 200
    wandb: bool = False
    slurm_id : int = 999
    out_dir : str = "models/VGG_blending_sweep"
    poison_type : str = "noise" #can choosen "badnets" or "blending" or "badnets_random"
    blending_alpha : float = 0.2
    poison_frac : float = 0.05
    clean_thresh : float = 0.77
    posion_thresh : float = 0.99
    _reproducible = True
    _seed : int = -1 
    _debug : bool = False
    _dim : tuple = (3,32,32)

try:
    args = dq.parse_args()
    cfg = Train_Config(**vars(args))
except:
    print("WARNING: USING DEFAULT CONFIGURATION, SLURM_ID=999")
    #terminate program if no arguments are passed
    cfg = Train_Config()
    
os.makedirs(f"./{cfg.out_dir}/models_pt", exist_ok=True)
#os.makedirs(f"./{cfg.out_dir}/models_np", exist_ok=True)
os.makedirs(f"./{cfg.out_dir}/metadata", exist_ok=True)
    
print(f"Training config: {cfg}")

transform = transforms.ToTensor()

# %%
cfg._seed = 42
cifar10_trainset = CIFAR10(root='./data', train=True, download=True, transform=None)
cifar10_testset = CIFAR10(root='./data', train=False, download=True, transform=None)
cfg._dim = cifar10_trainset.data.shape[1:3]


cifar10_trainset_poisoned, mask_train = gen_poison(cifar10_trainset, torch.arange(5), 6, cfg=cfg) 
cifar10_testset_poisoned, mask_test = gen_poison(cifar10_testset, torch.arange(len(cifar10_testset)), 6, cfg=cfg)
# %%
dq.grid(cifar10_testset_poisoned.data[:16],cifar10_testset_poisoned.targets[:16])
# %%