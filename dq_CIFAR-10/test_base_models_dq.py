# %%
import numpy as np

from torch import optim
from tqdm import tqdm
import os
from utils.model import CNN_classifier
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
import dq_poison
#import CIFAR10
USE_CUDA =True
device = torch.device("cuda" if USE_CUDA  and torch.cuda.is_available() else "cpu")
# %%


model = CNN_classifier(**asdict(dq.cnn_cfg)).to(device)
# %% 

# Training configuration and dataloader setup

# choose poisons 
# "badnets" or "blending" or "badnets_random" or "ulp_trainval" or "ulp_test"

@dataclass
class Test_Config:
    model_dir : str = "models/VGG_replicate/poison/trainval"
    metadata : str = "models/VGG_replicate/metadata_poison_trainval.csv"
    poison_type : str = "blending" #can choosen "badnets" or "blending" or "badnets_random" or 
    clean_thresh : float = 0.77
    posion_thresh : float = 0.99
    blending_alpha : float = 0.25
    _debug : bool = True
    _seed : int = -1
    wandb : bool = False
    mask_dir : str = "ulp_masks"
    mask : str = "mask00.bmp"

try:
    args = dq.parse_args()
    cfg = Test_Config(**vars(args))
except:
    print("WARNING: USING DEFAULT CONFIGURATION, SLURM_ID=999")
    #terminate program if no arguments are passed
    cfg = Test_Config()
    
    
print(f"Training config: {cfg}")

transform = transforms.ToTensor()

# %%
#cifar10_trainset = CIFAR10(root='./data', train=True, download=True, transform=None)
cifar10_testset = CIFAR10(root='./data', train=False, download=True, transform=None)
cifar10_testset_gpu = dq.GPUDataset(cifar10_testset, transform=transform)

all_metadata = dq.csv_reader(cfg.metadata, key_column="model_name")

model = CNN_classifier(**asdict(dq.cnn_cfg)).to(device)

# load bmp as tensor

masks = dq.load_images_to_dict(cfg.mask_dir)

# %%
runner = tqdm(list(all_metadata.items()))
worst_clean, worst_poison = 1 , 1

logfile = open("test_base_models_dq.log", "w")

data = dq.csv_reader(cfg.metadata, key_column="model_name")
SAMPLE_ENSEMBLE_SIZE = 128
model_paths = [] 
targets = []
for model_name, model_info in tqdm(list(data.items())):
    if model_info.target == "7":
        model_paths.append(os.path.join(cfg.model_dir, model_name) + ".pt")
        targets.append(int(model_info.target))
        
    if len(model_paths) >= SAMPLE_ENSEMBLE_SIZE:
        break

ensemble = dq.Ensemble(*dq.batch_load_models(model_paths)).to(device)
targets = torch.tensor(targets, device=device) # (num_models,)
# %%

#clean_acc, clean_loss = dq.evaluate_model(model, (cifar10_testset_gpu.data, cifar10_testset_gpu.targets))
best_poison = 0
for i in tqdm(range(50)):
    
    cfg._seed = i
    blending_params = dq_poison.gen_blend_params(cfg)
    
    poison_dataset = dq_poison.ulp(cifar10_testset,
                                  torch.arange(len(cifar10_testset)),
                                  poison_target=int(model_info.target),
                                  mask=torch.tensor(masks[model_info.mask]))
    
    # poison_dataset = dq_poison.blending(blending_params, 
    #                                     cifar10_testset, 
    #                                     range(len(cifar10_testset)),
    #                                     poison_target=int(model_info.target),
    #                                     cfg=cfg)
    
    poison_dataset_gpu = dq.GPUDataset(poison_dataset, transform=transform)
    poison_dataloader = DataLoader(poison_dataset_gpu, batch_size=512, shuffle=False)
    #ood_dataset_gpu = dq.GPUDataset(ood_poison_dataset, transform=transform)
    
    # model.load_state_dict(torch.load(os.path.join(cfg.model_dir, model_info.model_name) + ".pt"))
    # model.eval()
    # clean_acc, clean_loss = dq.evaluate_model(model, (cifar10_testset_gpu.data, cifar10_testset_gpu.targets))
    # poison_acc, poison_loss = dq.evaluate_model(model, (poison_dataset_gpu.data, poison_dataset_gpu.targets))
    # #ood_acc, ood_loss = dq.evaluate_model(model, (ood_dataset_gpu.data, ood_dataset_gpu.targets))
    # worst_clean = min(clean_acc, worst_clean)
    
    with torch.no_grad():
        ensemble_logits = ensemble(poison_dataset_gpu.data) # (num_models, batch_size, num_classes)
    
    
    ensemble_guess = torch.argmax(ensemble_logits, dim=-1) # (num_models, batch_size)
    correct = torch.sum(ensemble_guess == 7, dim=-1) / len(poison_dataset_gpu) # (num_models, batch_size)
    plt.hist(correct.detach().cpu().numpy())
    plt.title(f"ODD acc ULP on blending seed {i}")
    plt.show()
    
#     desc = f"{model_name=}, {clean_acc=}, {poison_acc=}, {worst_clean=}, {worst_poison=}"
#     runner.set_description(desc)
#     logfile.write(desc + "\n")
#     logfile.flush()

# logfile.close()
# %%


        
# %%