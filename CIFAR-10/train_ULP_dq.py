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
import einops
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
from einops.layers.torch import Rearrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%

# %%
# %%

# %%
@dataclass
class Train_Config:
    epochs: int = 20
    wandb_project: str = "ULP-CIFAR10"
    wandb: bool = True
    clean_train_dir : str = "new_models/clean/trainval/*.pt"
    clean_test_dir : str = "new_models/clean/trainval/*.pt"
    poison_train_dir : str = "new_models/poison/trainval/*.pt"
    poison_test_dir : str = "new_models/poison/test/*.pt"
    
    # clean_dir : str = "new_models/clean/models_pt/*.pt"
    # poison_train_dir : str = "new_models/poison_train/models_pt/*.pt"
    # poison_test_dir : str = "new_models/poison_test/models_pt/*.pt"
    num_train : int = 4000
    num_test : int = 500
    #====================================
    
    num_ulps: int = 10
    meta_lr : float = 1e-3
    ulp_lr : float = 1e2 #WTF LR=100?
    ulp_scale : float = 1
    tv_reg : float = 1e-6
    meta_bs : int = 100
    grad_clip_threshold: float = None  # Set a default value
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

print(f"Config: {asdict(cfg)}")
    
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
    clean_models_test = raw_clean_models_train[cfg.num_train // 2: cfg.num_train//2 + cfg.num_test // 2]
else:
    clean_models_test = raw_clean_models_test[:cfg.num_test // 2]

poisoned_models_train = raw_poisoned_models_train[:cfg.num_train // 2]

if cfg.poison_test_dir == cfg.poison_train_dir:
    poisoned_models_test = raw_poisoned_models_train[cfg.num_train // 2 : cfg.num_train // 2 + cfg.num_test // 2]
else:
    poisoned_models_test = raw_poisoned_models_test[:cfg.num_test // 2]

models_train_paths = np.array(clean_models_train + poisoned_models_train)
labels_train = torch.tensor([0]*len(clean_models_train) + [1]*len(poisoned_models_train), dtype=torch.long, device=device)

models_test_paths = np.array(clean_models_test + poisoned_models_test)
labels_test = torch.tensor([0]*len(clean_models_test) + [1]*len(poisoned_models_test), dtype=torch.long, device=device)

initial_mem = torch.cuda.memory_allocated(device)

def batch_load_models(paths):
    models = []
    runner = tqdm(paths)
    for path in runner:
        model_state = torch.load(path, map_location=device)
        model = CNN_classifier(**asdict(dq.cnn_cfg))
        model.load_state_dict(model_state)
        models.append(model.to(device))
        runner.set_description(f"Loading {path}")
    return np.array(models)

models_train = batch_load_models(models_train_paths)

batch_mem = torch.cuda.memory_allocated(device) - initial_mem
torch.cuda.empty_cache() # Clear unused memory

models_test = batch_load_models(models_test_paths)
test_ensemble = dq.Ensemble(*models_test).to(device)
test_ensemble.eval()
test_mem = torch.cuda.memory_allocated(device) - initial_mem

print(f"Memory used by batch models: {batch_mem/(1024**3):.2f} GBs")
print(f"Memory used by test models: {test_mem/(1024**3):.2f} GBs")

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

model = CNN_classifier(**asdict(dq.cnn_cfg)) #3.77Mb
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

meta_classifier = nn.Sequential(
    Rearrange('b u c -> b (u c)'),
    nn.Linear(cfg.num_ulps * dq.cnn_cfg.nofclasses, 2)
).to(device)
    
ULPs=torch.rand((cfg.num_ulps,3,32,32),device=device) * cfg.ulp_scale
#ULPs = torch.load("ulp_rigged.pt").to(device)
ULPs = nn.Parameter(ULPs, requires_grad=True)              #1e+2


# %%
    # Initialize wandb
if cfg.wandb:
    wandb.init(project=cfg.wandb_project, config = asdict(cfg))
        

# %%



opt_meta = optim.Adam(meta_classifier.parameters(), lr=cfg.meta_lr)   #1e-3
opt_ULPs = optim.SGD(params=[ULPs],lr=cfg.ulp_lr)   

best_acc = 0
train_accuracy = 0
test_accuracy = 0
criterion=torch.nn.CrossEntropyLoss()
runner = tqdm(range(cfg.epochs))
for epoch in runner:
    
    correct = 0
    count = 0
    train_loss = 0
    
    for i, (model_idx, labels) in enumerate(train_loader):
        
        train_ensemble = dq.Ensemble(*models_train[model_idx]).to(device)
        train_ensemble.eval()
        
        # model_batch = train_models[model_idx]
        # ensemble = dq.Ensemble(*model_batch)
        # ensemble.to(device)
        # ensemble.eval()
        model_logits = train_ensemble(ULPs, average=False, split=False) #(BS, ULP, 10) -> (BS, ULP*10)?
        meta_logits = meta_classifier(model_logits) # (BS, 2)
        reg_loss = cfg.tv_reg * tv_norm(ULPs)
        base_loss = criterion(meta_logits,labels)
        
        y_guess = torch.argmax(meta_logits, dim=1)
        batch_correct = torch.sum(y_guess == labels).item()
        correct += batch_correct
        count += len(labels)
        
        loss = base_loss + reg_loss
        train_loss += loss.item()
        opt_ULPs.zero_grad()
        opt_meta.zero_grad()

        loss.backward()

        if cfg.grad_clip_threshold is not None:
            torch.nn.utils.clip_grad_norm_(ULPs, cfg.grad_clip_threshold)

        opt_ULPs.step()
        opt_meta.step()

        # Keep ULP in range [0,1]
        torch.clamp_(ULPs.data, 0, cfg.ulp_scale)



        runner.set_description(f"batch={i+1}/{len(train_loader)}, loss={loss.item():.4f}, reg_loss={reg_loss.item():.4f}, base_loss={base_loss.item():.4f}, Train Acc={train_accuracy:.4f}, Test Acc={test_accuracy:.4f}")
        
        grad_norm_ULPs = torch.norm(ULPs.grad)
        grad_norm_meta_classifier = torch.norm(torch.cat([p.grad.view(-1) for p in meta_classifier.parameters() if p.grad is not None]))

        batch_acc = batch_correct / len(labels)

        batch_stats = {
            "train_loss": loss.item(),
            "reg_loss": reg_loss.item(),
            "base_loss": base_loss.item(),
            "batch_acc": batch_acc,
            "epoch": epoch+1,
            "grad_norm_ULPs": grad_norm_ULPs.item(),
            "grad_norm_meta_classifier": grad_norm_meta_classifier.item()
        }
        
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
        "epoch" : epoch+1
    }
    
    best_acc = max(best_acc, test_accuracy)
    if best_acc < 0.60 and epoch > 20: #give up
        break

    if cfg.wandb:
        wandb.log(epoch_stats)
        
    # Plot ULPs and histogram
    display.clear_output(wait=True)
    display.display(plt.gcf())
    dq.grid(ULPs.data)
    if cfg.wandb:
        wandb.log({"ULPs": [wandb.Image(img) for img in torch.unbind(ULPs.data)]})
    
    pixel_values = ULPs.detach().cpu().numpy().flatten()  # Flatten the ULPs to get a 1D array of pixel values
    plt.hist(pixel_values, bins=100)
    # Create a histogram using matplotlib
    histogram = wandb.Histogram(pixel_values)

    # Log the histogram to wandb
    if cfg.wandb:
        wandb.log({"ULP Pixel Value Distribution": histogram})
            
# %%   
        #wandb.Histogram(np_histogram = np_hist)
        # Log the histogram to wandb
# save ULPs and meta_classifier under folder based on run name
if cfg.wandb:
    base_dir = "wandb_artifacts"
    os.makedirs(base_dir, exist_ok=True)
    run_id = wandb.run.id
    # Save ULPs and meta_classifier
    torch.save(ULPs, f'./{base_dir}/ULPs_{run_id}.pth')
    torch.save(meta_classifier.state_dict(), f'./{base_dir}/meta_classifier_{run_id}.pth')
    # Create a new artifact for ULPs
    ulps_artifact = wandb.Artifact('ULPs', type='model')
    ulps_artifact.add_file(f'./{base_dir}/ULPs_{run_id}.pth')

    # Create a new artifact for meta_classifier
    meta_classifier_artifact = wandb.Artifact('meta_classifier', type='model')
    meta_classifier_artifact.add_file(f'./{base_dir}/meta_classifier_{run_id}.pth')

    # Log the artifacts
    wandb.log_artifact(ulps_artifact)
    wandb.log_artifact(meta_classifier_artifact)

        
if cfg.wandb:
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
