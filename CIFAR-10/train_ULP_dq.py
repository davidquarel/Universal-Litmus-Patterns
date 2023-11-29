# %%
# # Universal Patterns for Revealing Backdoors CIFAR-10
# 
# Here we perform our optimization to obtain the universal pattern that help us reveal the backdoor.

import numpy as np
import torch
from torch import optim

from utils.model import CNN_classifier

import pickle
import time
import glob
from tqdm import tqdm

import os
import sys

import torch
from torch.utils import data
import torch.nn as nn
import logging
import dq 
from dataclasses import dataclass, asdict, fields
from torch.utils.data import Dataset, DataLoader
import argparse
import wandb
import pickle
from IPython import display
import matplotlib.pyplot as plt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%

# %%
# %%

# %%
@dataclass
class Train_Config:
    epochs: int = 1000
    wandb_project: str = "ULP-CIFAR10-2"
    wandb: bool = True
    clean_train_dir : str = "old_models/clean/trainval/*.pt"
    clean_test_dir : str = "old_models/clean/test/*.pt"
    poison_train_dir : str = "old_models/poison/trainval/*.pt"
    poison_test_dir : str = "old_models/poison/test/*.pt"
    
    # clean_dir : str = "new_models/clean/models_pt/*.pt"
    # poison_train_dir : str = "new_models/poison_train/models_pt/*.pt"
    # poison_test_dir : str = "new_models/poison_test/models_pt/*.pt"
    num_train : int = 1000
    num_test : int = 200
    #====================================
    num_ulps: int = 10
    meta_lr : float = 1e-3
    ulp_lr : float = 1e3 #WTF LR=100? 
    tv_reg : float = 1e-5
    meta_bs : int = 50
    grad_clip_threshold: float = 0  # Set a default value
    hyper_param_search: bool = False

# %%
cfg = Train_Config()


def parse_args():
    parser = argparse.ArgumentParser(description="Training Configuration")
    
    # Dynamically add arguments based on the dataclass fields
    for field in fields(Train_Config):
        if field.type == bool:
            # For boolean fields, use 'store_true' or 'store_false'
            parser.add_argument(f'--{field.name}', action='store_true' if not field.default else 'store_false')
        else:
            parser.add_argument(f'--{field.name}', type=field.type, default=field.default, help=f'{field.name} (default: {field.default})')

    args = parser.parse_args()
    return args

try:
    args = parse_args()
    cfg = Train_Config(**vars(args))
except:
    print("WARNING: USING DEFAULT CONFIGURATION, SLURM_ID=999")
    #terminate program if no arguments are passed
    cfg = Train_Config()

    
# %%

torch.cuda.empty_cache() # Clear unused memory

raw_clean_models_train = sorted(glob.glob(cfg.clean_train_dir))
raw_poisoned_models_train = sorted(glob.glob(cfg.poison_train_dir))
raw_clean_models_test = sorted(glob.glob(cfg.clean_test_dir))
raw_poisoned_models_test = sorted(glob.glob(cfg.poison_test_dir))

print(f"Found {len(raw_clean_models_train)} clean train models")
print(f"Found {len(raw_poisoned_models_train)} poisoned train models")
print(f"Found {len(raw_clean_models_test)} clean test models")
print(f"Found {len(raw_poisoned_models_test)} poisoned test models")

clean_models_train = raw_clean_models_train[:cfg.num_train // 2]

if cfg.clean_test_dir == cfg.clean_train_dir:
    clean_models_test = raw_clean_models_train[cfg.num_train // 2:cfg.num_train]
else:
    clean_models_test = raw_clean_models_test[:cfg.num_test // 2]

poisoned_models_train = raw_poisoned_models_train[:cfg.num_train // 2]

if cfg.poison_test_dir == cfg.poison_train_dir:
    poisoned_models_test = raw_poisoned_models_train[cfg.num_train // 2:cfg.num_train]
else:
    poisoned_models_test = raw_poisoned_models_test[:cfg.num_test // 2]

models_train = np.array(clean_models_train + poisoned_models_train)
labels_train = torch.tensor([0]*len(clean_models_train) + [1]*len(poisoned_models_train), dtype=torch.long, device=device)

models_test = np.array(clean_models_test + poisoned_models_test)
labels_test = torch.tensor([0]*len(clean_models_test) + [1]*len(poisoned_models_test), dtype=torch.long, device=device)

initial_mem = torch.cuda.memory_allocated(device)

train_models = np.array([CNN_classifier(**asdict(dq.cnn_cfg)).to(device) for _ in tqdm(range(len(models_train)),desc="Init batch ensemble")])

batch_mem = torch.cuda.memory_allocated(device) - initial_mem
torch.cuda.empty_cache() # Clear unused memory

test_models = [CNN_classifier(**asdict(dq.cnn_cfg)).to(device) for _ in tqdm(range(len(models_test)), desc="Init test ensemble")]
test_ensemble = dq.Ensemble(*test_models).to(device)
test_ensemble.eval()
test_mem = torch.cuda.memory_allocated(device) - initial_mem

print(f"Memory used by batch models: {batch_mem/(1024**2)} MBs")
print(f"Memory used by test models: {test_mem/(1024**2)} MBs")

print(f"Clean train models: {len(clean_models_train)}")
print(f"Poisoned train models: {len(poisoned_models_train)}")
print(f"Clean test models: {len(clean_models_test)}")
print(f"Poisoned test models: {len(poisoned_models_test)}")

# print warning if number of models doesn't match cfg.num_train or cfg.num_test
if len(models_train) != cfg.num_train:
    print(f"WARNING: cfg.num_train={cfg.num_train}, but {len(models_train)} models loaded")
if len(models_test) != cfg.num_test:
    print(f"WARNING: cfg.num_test={cfg.num_test}, but {len(models_test)} models loaded")


# %%
    
class IdxDataset(Dataset):
    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        """
        Return the number of items in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Load and return a model and its label from a given index.
        :return: A tuple (model, label)
        """
        return idx, self.labels[idx] 
    
train_idx = IdxDataset(labels_train)
train_loader = DataLoader(train_idx, batch_size=cfg.meta_bs, shuffle=True)
    
# %% 
# ====================================

def tv_norm(x):
    return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
        torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))    


# %%

# %%
# val_models = []

# def paths_to_models(paths):
#     models = []
#     runner = tqdm(paths)
#     for path in runner:
#         model_state = torch.load(path)
#         model = CNN_classifier(**asdict(dq.cnn_cfg)).to(device)
#         model.load_state_dict(model_state)
#         models.append(model)
#         runner.set_description(f"Loading {path}")
#     return np.array(models) 

# train_models = paths_to_models(model_paths_train)
# val_models = paths_to_models(model_paths_val)
# test_ensemble = dq.Ensemble(*val_models)

# %%

def train():
    # ### Perform Optimization
    if cfg.hyper_param_search:
        cfg.meta_lr = 10 ** random.uniform(-4, -2)  # Log-uniform between 1e-4 and 1e-2
        cfg.ulp_lr = 10 ** random.uniform(2, 4)     # Log-uniform between 1e2 and 1e4
        cfg.tv_reg = 10 ** random.uniform(-7, -5)   # Log-uniform between 1e-7 and 1e-5
        cfg.meta_bs = random.choice([16, 32, 64, 128])
        cfg.grad_clip_threshold = random.choice([0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0])

        # Initialize wandb
    wandb.init(project=cfg.wandb_project, config={
        "meta_lr": cfg.meta_lr,
        "ulp_lr": cfg.ulp_lr,
        "tv_reg": cfg.tv_reg,
        "meta_bs": cfg.meta_bs,
        "grad_clip_threshold": cfg.grad_clip_threshold
    })
        
    
    meta_classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(cfg.num_ulps * dq.cnn_cfg.nofclasses, 2)
    ).to(device)

    opt_meta = optim.Adam(meta_classifier.parameters(), lr=cfg.meta_lr)   #1e-3


        
    ULPs=torch.rand((cfg.num_ulps,3,32,32),device=device)
    ULPs = nn.Parameter(ULPs, requires_grad=True)
    opt_ULPs = optim.SGD(params=[ULPs],lr=cfg.ulp_lr)                 #1e+2
    # %%

    # %%
    test_accuracy = 0
    train_accuracy = 0
    train_loss = 0
    criterion=torch.nn.CrossEntropyLoss()
    runner = tqdm(range(cfg.epochs))
    for epoch in runner:
        
        correct = 0
        count = 0
        
        for i, (model_idx, labels) in enumerate(train_loader):
            
            train_ensemble = dq.Ensemble(*train_models[model_idx]).to(device)
            train_ensemble.eval()
            
            # model_batch = train_models[model_idx]
            # ensemble = dq.Ensemble(*model_batch)
            # ensemble.to(device)
            # ensemble.eval()
            
            model_logits = train_ensemble(ULPs, average=False, split=False) #(BS, ULP, 10)
            classifier_logits = meta_classifier(model_logits)
            reg_loss = cfg.tv_reg * tv_norm(ULPs)
            base_loss = criterion(classifier_logits,labels)
            
            y_guess = torch.argmax(classifier_logits, dim=1)
            correct += torch.sum(y_guess == labels).item()
            count += len(labels)
            
            loss = base_loss + reg_loss
            train_loss += loss.item()
            opt_ULPs.zero_grad()
            opt_meta.zero_grad()

            loss.backward()

            if cfg.grad_clip_threshold != 0:
                torch.nn.utils.clip_grad_norm_(ULPs, cfg.grad_clip_threshold)

            opt_ULPs.step()
            opt_meta.step()

            # Keep ULP in range [0,1]
            torch.clamp_(ULPs.data, 0, 1)

            batch_stats = {
                "train_loss": loss.item(),
                "reg_loss": reg_loss.item(),
                "base_loss": base_loss.item(),
            }

            runner.set_description(f"batch={i+1}/{len(train_loader)}, loss={loss.item():.4f}, reg_loss={reg_loss.item():.4f}, base_loss={base_loss.item():.4f}, Train Acc={train_accuracy:.4f}, Test Acc={test_accuracy:.4f}")
            
            grad_norm_ULPs = torch.norm(ULPs.grad)
            grad_norm_meta_classifier = torch.norm(torch.cat([p.grad.view(-1) for p in meta_classifier.parameters() if p.grad is not None]))

            # Log gradient norms
            batch_stats.update({
                "grad_norm_ULPs": grad_norm_ULPs.item(),
                "grad_norm_meta_classifier": grad_norm_meta_classifier.item(),
                "model_indices": model_idx.cpu().numpy().tolist(),
            })
            
            if cfg.wandb:
                wandb.log(batch_stats)
        
        with torch.no_grad():
            # Evaluate on train set
            model_logits = test_ensemble(ULPs, average=False, split=False) #(100, ULP, 10)
            meta_logits = meta_classifier(model_logits)
            y_guess = torch.argmax(meta_logits, dim=1)
            test_accuracy = torch.sum(y_guess == labels_test).item() / len(y_guess)
            
        train_accuracy = correct / count    
        train_loss /= len(train_loader)
        epoch_stats = {
            "train_acc": train_accuracy,
            "test_acc": test_accuracy,
            "train_loss": train_loss,
        }
            
        if cfg.wandb:
            wandb.log(epoch_stats)
            
        # Plot ULPs and histogram
        if epoch % 5 == 4:
            display.clear_output(wait=True)
            display.display(plt.gcf())
            dq.grid(ULPs.data)
            wandb.log({"ULPs": [wandb.Image(img) for img in torch.unbind(ULPs.data)]})
            
            pixel_values = ULPs.detach().cpu().numpy().flatten()  # Flatten the ULPs to get a 1D array of pixel values
            # Create a histogram using matplotlib
            histogram = wandb.Histogram(pixel_values)

            # Log the histogram to wandb
            wandb.log({"ULP Pixel Value Distribution": histogram})
            
            
            #wandb.Histogram(np_histogram = np_hist)
            # Log the histogram to wandb
    wandb.finish()
# %%

    

    
    
# %%

# def paper_ULPs():

#     with open("./results/ULP_vggmod_CIFAR-10_N10.pkl", "rb") as f:
#         ulp10 = pickle.load(f)
#         ULPs = ulp10[0]
#         W = ulp10[1]
#         b = ulp10[2]

#     ULPs = ulp10[0] / 255
#     with torch.no_grad():
#         # Evaluate on train set
#         model_logits = test_ensemble(ULPs, average=False, split=False) #(100, ULP, 10)
#         meta_classifier[1].weight.data = W.T
#         meta_classifier[1].bias.data = b
#         meta_logits = meta_classifier(model_logits)
#         y_guess = torch.argmax(meta_logits, dim=1)
#         val_accuracy = torch.sum(y_guess == labels_val).item() / len(y_guess)
#         print(f"Val accuracy: {val_accuracy:.5f}")
        
#     dq.grid(ULPs)
    

# %%