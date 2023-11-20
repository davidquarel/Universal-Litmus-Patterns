

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
import random
import torch
import torchvision
import torchvision.transforms as transforms
import glob
import matplotlib.pyplot as plt
import wandb
import einops

import dq
from dq import GPUDataset, evaluate_model, trainer, grid, peek

#import CIFAR10
USE_CUDA =True
device = torch.device("cuda" if USE_CUDA  and torch.cuda.is_available() else "cpu")

import re
import pickle

def extract_id(string):
    # Extract the four-digit number from the string using a regular expression
    match = re.search(r'\d{4}', string)
    # If an ID is found, convert it to an integer and return it
    if match:
        return int(match.group())
    # Return None if no ID was found
    return None


def get_shuffled_data(path):
    all_data = sorted(glob.glob(path))
    
    return all_data

attacked_data_test = sorted(glob.glob(f'Attacked_Data/test/*.pkl'))
poisoned_models_test = sorted(glob.glob(f'poisoned_models/test/*.pt'))


test_masks = ["mask01.bmp", "mask02.bmp", "mask2.bmp", "mask09.bmp", "mask7.bmp", "mask1.bmp", "mask03.bmp", "mask4.bmp", "mask04.bmp", "mask06.bmp"]
trainval_masks = ["mask8.bmp", "mask6.bmp", "mask10.bmp", "mask08.bmp", "mask9.bmp", "mask05.bmp", "mask07.bmp", "mask5.bmp", "mask00.bmp", "mask3.bmp"]

poisoned_paths = ["poisoned_models/test", "poisoned_models/trainval"]

# %%
import csv
from dataclasses import dataclass


import csv
from dataclasses import dataclass

@dataclass
class ModelInfo:
    mask: str
    confidence: float
    source_class: int
    target_class: int
    backdoor_file: str

def load_data_into_dict(csv_file):
    data_dict = {}
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            model_info = ModelInfo(
                mask=row['trigger pattern'],
                confidence=float(row['trigger pattern confidence']),
                source_class=int(row['source class']),
                target_class=int(row['target class']),
                backdoor_file=row['backdoor data file']
            )
            data_dict[row['model file path']] = model_info
    return data_dict
# Example usage
# %%

metadata = load_data_into_dict('results/poisoned_models_metadata.csv')
# %%


def run_poison_acc():
    model_stats = {}
    metadata_path = 'results/poisoned_models_metadata.csv'  # Replace with your CSV file path
    model = CNN_classifier(**asdict(dq.cnn_cfg)).to(device)
    with open("results/model_poisonacc_matched.csv", "w") as f:
        f.write("model_path,accuracy,test_loss,source,target,mask\n")
        for dataset in ["test", "trainval"]:
            all_models = sorted(glob.glob(f'poisoned_models/{dataset}/*.pt'))
            all_metadata = load_data_into_dict(metadata_path)            
            runner = tqdm(all_models)
            for (i, model_path) in enumerate(runner):
                #check ID match
                #assert extract_id(data_path) == extract_id(model_path) == i
                backdoor_file = all_metadata[model_path].backdoor_file
                mask = all_metadata[model_path].mask
                source = all_metadata[model_path].source_class
                target = all_metadata[model_path].target_class
                
                with open(backdoor_file, 'rb') as file:
                    X_poisoned, Y_poisoned, trigger, source_l, target_l = pickle.load(file)
                    assert source == source_l and target == target_l
                    X_poisoned = einops.rearrange(X_poisoned, 'b h w c -> b c h w')
                    X_poisoned = torch.from_numpy(X_poisoned).type(torch.FloatTensor).contiguous().to(device)
                    Y_poisoned = torch.from_numpy(Y_poisoned).to(device)
                
                model.load_state_dict(torch.load(model_path))
                model.eval()
                # make it a float tensor
                
                logits = model(X_poisoned)
                Y_guess = torch.argmax(logits, dim=1)
                accuracy = torch.sum(Y_guess == target).item() / len(Y_guess)
                test_loss = torch.nn.CrossEntropyLoss()(logits, Y_poisoned).item()
                
                
                base_name = os.path.basename(model_path)
                model_stats[base_name] = accuracy
                runner.set_description(f"{model_path} acc {accuracy:.5f} test_loss {test_loss:.5f} source {source} target {target} mask {mask}")
                
                f.write(f"{model_path},{accuracy},{test_loss}, {source},{target},{mask}\n")
                f.flush()
    f.close()
    return model_stats
# %%
#poison_model_stats = run_poison_acc()
# %%
with open(attacked_data_test[0], 'rb') as file:
    X_poisoned, Y_poisoned, trigger, source, target = pickle.load(file)
    X_poisoned = einops.rearrange(X_poisoned, 'b h w c -> b c h w')
    X_poisoned = torch.from_numpy(X_poisoned).type(torch.FloatTensor).contiguous().to(device)
    grid(X_poisoned[:16])
# %%
#poison_model_stats = run_poison_acc()
# %%
