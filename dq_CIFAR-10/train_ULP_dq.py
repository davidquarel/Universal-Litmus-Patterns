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
import glob
from collections import namedtuple
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%

    
# %%

def split_models(train_paths, test_paths, cfg=None):
    split_train = train_paths[:cfg.num_train // 2]

    if train_paths == test_paths:
        split_test = train_paths[cfg.num_train // 2: cfg.num_train//2 + cfg.num_test // 2]
    else:
        split_test = test_paths[:cfg.num_test // 2]
        
    #check there is no overlap between test and train
    assert (len(set(split_train).intersection(set(split_test))) == 0, "Models overlap between train and test")
    return split_train, split_test
    
def generate_test_ensembles(clean_models, poisons_models={}, cfg=None):
    """
    Generates test ensembles from the given model sets.
    """
    CLEAN_LABEL = 0
    POISON_LABEL = 1
    
    test_ensembles = {
        "test_clean_acc":  (dq.Ensemble(*dq.batch_load_models(np.array(clean_models))).to(device), CLEAN_LABEL),
    }
    
   
    for ood_name, ood_dir in poisons_models.items():
        ood_models = sorted(glob.glob(f"{ood_dir}/*{cfg._model_ext}"))[-(cfg.num_test // 2):]
        test_ensembles[f"test_{ood_name}_acc"] = (dq.Ensemble(*dq.batch_load_models(np.array(ood_models))).to(device), POISON_LABEL)
    
    return test_ensembles


def init_models_train(cfg):

    # Define directories and their corresponding labels
    model_directories = {
        'clean_train': cfg.clean_train_dir,
        'poisoned_train': cfg.poison_train_dir,
        'clean_test': cfg.clean_test_dir if cfg.clean_test_dir is not None else cfg.clean_train_dir,
        'poisoned_test': cfg.poison_test_dir if cfg.poison_test_dir is not None else cfg.poison_train_dir,
    }

    # Read model paths for each category
    model_paths = {}
    
    for label, directory in model_directories.items():
        if directory is None:
            continue
        model_paths[label] = sorted(glob.glob(f"{directory}/*{cfg._model_ext}"))

    # Print counts
    for label, paths in model_paths.items():
        print(f"Found {len(paths)} models >=< {label}")

    # Splitting models into train and test
    clean_models_train, clean_models_test = split_models(model_paths['clean_train'], model_paths['clean_test'], cfg=cfg)
    poisoned_models_train, poisoned_models_test = split_models(model_paths['poisoned_train'], model_paths['poisoned_test'], cfg=cfg)
    

    # check there is no overlap between test and train
    assert (len(set(clean_models_train).intersection(set(clean_models_test))) == 0, 
            "Clean models overlap between train and test")
    assert (len(set(poisoned_models_train).intersection(set(poisoned_models_test))) == 0, 
        "Poisoned models overlap between train and test")
    
    models_train_paths = np.array(clean_models_train + poisoned_models_train)
    labels_train = torch.tensor([0]*len(clean_models_train) + [1]*len(poisoned_models_train), 
                                dtype=torch.long, device=device)
    
    return models_train_paths, labels_train, clean_models_test, poisoned_models_test




# %%

def post_ULP(x, sigmoid_no_clip=False):
    return torch.sigmoid(x) if sigmoid_no_clip else x

def eval_ensemble(test_ensembles, ulps, meta_classifier, cfg=None):
    
    test_stats = {}
    
    for name, (ensemble, target) in test_ensembles.items():
        with torch.no_grad():
            # Evaluate on train set
            ensemble.eval()
            model_logits = ensemble(post_ULP(ulps, cfg.sigmoid_no_clip), average=False, split=False) #(100, ULP, 10)
            meta_logits = meta_classifier(model_logits)
            y_guess = torch.argmax(meta_logits, dim=1)
            acc = torch.sum(y_guess == target).item() / len(y_guess)
        test_stats[name] = acc
    return test_stats


# %%

def init_ULP_and_meta_classifier(cfg):
    meta_classifier = nn.Sequential(
        Rearrange('b u c -> b (u c)'),
        nn.Linear(cfg.num_ulps * dq.cnn_cfg.nofclasses, 2)
    ).to(device)
        
    if cfg.sigmoid_no_clip:
        print("WARNING: Using sigmoid on ULPs, init from N(0,1)")
        ULPs = torch.randn((cfg.num_ulps,3,32,32),device=device) * cfg.ulp_scale
    else:
        print("WARNING: Not using sigmoid on ULPs, init from U(0,1)")
        ULPs = torch.rand((cfg.num_ulps,3,32,32),device=device) * cfg.ulp_scale

    #ULPs = torch.load("ulp_rigged.pt").to(device)
    ULPs = nn.Parameter(ULPs, requires_grad=True)              #1e+2
    
    return ULPs, meta_classifier
    
def log_artifacts(artifacts, cfg=None):
    base_dir = f"artifacts/{cfg.wandb_project}"
    os.makedirs(base_dir, exist_ok=True)
    run_id = wandb.run.id
    # Save ULPs and meta_classifier
    for name, artifact in artifacts.items():
        torch.save(artifact, f'./{base_dir}/{name}_{run_id}.pth')
        name_artifact = wandb.Artifact(name, type='model')
        name_artifact.add_file(f'./{base_dir}/{name}_{run_id}.pth')
        wandb.log_artifact(name_artifact)
        
# %% 
        

def main(ULPs, meta_classifier, models_train, labels_train, test_ensembles, cfg=None):
    
    train_loader = DataLoader(dq.IdxDataset(labels_train), batch_size=cfg.meta_bs, shuffle=True)
    
    best_acc = -1
    opt_meta = optim.Adam(meta_classifier.parameters(), lr=cfg.meta_lr)   #1e-3
    opt_ULPs = optim.SGD(params=[ULPs],lr=cfg.ulp_lr)   
    epoch_stats = {}
    criterion=torch.nn.CrossEntropyLoss()
    runner = tqdm(range(cfg.epochs))

    for epoch in runner:
        
        correct = 0
        count = 0
        train_loss = 0
        
        for i, (model_idx, labels) in enumerate(train_loader):
            
            train_ensemble = dq.Ensemble(*models_train[model_idx]).to(device)
            train_ensemble.eval()
            
            model_logits = train_ensemble(post_ULP(ULPs, cfg.sigmoid_no_clip), average=False, split=False) #(BS, ULP, 10) -> (BS, ULP*10)?
            meta_logits = meta_classifier(model_logits) # (BS, 2)
            reg_loss = cfg.tv_reg * dq.tv_norm(ULPs)
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
            if not cfg.sigmoid_no_clip:
                torch.clamp_(ULPs.data, 0, cfg.ulp_scale)


            runner_info = dq.render_runner_info(i, len(train_loader), epoch_stats)
            runner.set_description(runner_info)
            
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
        
        test_stats = eval_ensemble(test_ensembles, post_ULP(ULPs, cfg.sigmoid_no_clip), meta_classifier, cfg=cfg)
            
        train_accuracy = correct / count    
        train_loss /= len(train_loader)
        # test_accuracy = (test_stats["test_clean_acc"] + test_stats["test_poison_acc"]) / 2
        
        epoch_stats = {
            "train_acc": train_accuracy,
            #"test_acc": test_accuracy,
            "train_loss": train_loss,
            "epoch" : epoch+1
        }
        epoch_stats.update(test_stats)
        
        # best_acc = max(best_acc, test_accuracy)
        # if best_acc < cfg.acc_thresh and epoch > cfg.epoch_thresh: #give up
        #     break

   
        # # Plot ULPs and histogram
        if cfg._debug:
            display.clear_output(wait=True)
            display.display(plt.gcf())
            dq.grid(ULPs.data)
                 # Create a histogram using matplotlib
        # Log the histogram to wandb
        if cfg.wandb:
            pixel_values = ULPs.detach().cpu().numpy().flatten()  # Flatten the ULPs to get a 1D array of pixel values
            plt.hist(pixel_values, bins=100)
            histogram = wandb.Histogram(pixel_values)
            wandb.log({"ULP Pixel Value Distribution": histogram})
            
            wandb.log(epoch_stats)
            wandb.log({"ULPs": [wandb.Image(img) for img in torch.unbind(torch.sigmoid(ULPs.data))]})
            log_artifacts({"ULPs": ULPs.data, "meta_classifier": meta_classifier.state_dict()}, cfg=cfg)
                
    if cfg.wandb:
        wandb.finish()  
    
# %%


@dataclass
class Train_Config:
    wandb_project: str = "ULP-CIFAR10-ood3"
    wandb: bool = True
    wandb_name : str = None
    wandb_desc : str = None
    #====================================
    epochs: int = 20
    clean_train_dir : str = "models/VGG_replicate/clean/trainval"
    clean_test_dir : str = None 
    poison_train_dir : str = "models/VGG_good_blended_alpha_25/models_pt"
    poison_test_dir : str = "models/VGG_good_blended_alpha_25/models_pt"
    _ood_test_dirs : str = None
    
    # clean_dir : str = "new_models/clean/models_pt/*.pt"
    # poison_train_dir : str = "new_models/poison_train/models_pt/*.pt"
    # poison_test_dir : str = "new_models/poison_test/models_pt/*.pt"
    num_train : int = 3000
    num_test : int = 500
    #====================================
    acc_thresh : float = -1 #dummy value model will always exceed
    epoch_thresh : int = 0
    num_ulps: int = 10
    meta_lr : float = 1e-3
    ulp_lr : float = 1e2 #WTF LR=100?
    ulp_scale : float = 1
    tv_reg : float = 1e-6
    meta_bs : int = 100
    sigmoid_no_clip : bool = False #if true, sigmoid the ULP and do not clip to [0,1]
    grad_clip_threshold: float = None  # Set a default value
    hyper_param_search: bool = False
    cache_dataset : bool = False #preload entire dataset into GPU memory
    #====================================
    _model_ext : str = ".pt"
    _debug : bool = False
    _poison_name : str = None

# %%
