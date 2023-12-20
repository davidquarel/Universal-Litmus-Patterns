# %%
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
import einops
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt 
# %%

test_img_idx = []
test_labels = []
found = 0

num_imgs = 5

for target in range(num_imgs):
    for i in range(len(dq.cifar10_testset)):
        if dq.cifar10_testset[i][1] == target:
            test_img_idx.append(i)
            test_labels.append(target)
            break

test_img_idx = np.array(test_img_idx)
test_labels = np.array(test_labels)

clean = dq.cifar10_testset.data[test_img_idx]
poison = clean.copy()

ulp = np.zeros((2 * num_imgs,32,32,3), dtype=np.uint8)
ulp[:num_imgs] = clean

# load the masks
path = "Masks/trainval"
path_mask = dq.load_images_to_numpy_arrays(path)
masks = np.array([x[1] for x in path_mask])

for i in range(num_imgs):
    poison[i, 1:1+5, 1:1+5] = masks[i]
ulp[num_imgs:] = poison

ulp = einops.rearrange(torch.from_numpy(ulp), 'b h w c -> b c h w').float().contiguous()
ulp_labels = torch.tensor(np.concatenate((test_labels, test_labels)))

dq.grid(ulp, ulp_labels)
# %%
# load model

def test_ulp(path):
    model = CNN_classifier(**asdict(dq.cnn_cfg)).to(device)
    model.load_state_dict(torch.load(path))
    logits = model(ulp.to(device))
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    plt.imshow(probs)
    plt.title(path)
    plt.show()
    
idx = 10
test_ulp(f"old_models/clean/test/clean_vggmod_CIFAR-10_{idx:04d}.pt")
test_ulp(f"old_models/clean/trainval/clean_vggmod_CIFAR-10_{idx:04d}.pt")
test_ulp(f"old_models/poison/test/poisoned_vggmod_CIFAR-10_{idx:04d}.pt")
test_ulp(f"old_models/poison/trainval/poisoned_vggmod_CIFAR-10_{idx:04d}.pt")
# %%
# save ulps to disk
torch.save(ulp, "ulp_rigged.pt")
# %%
