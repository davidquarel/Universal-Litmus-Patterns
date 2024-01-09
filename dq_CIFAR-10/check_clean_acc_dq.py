# %%
import numpy as np

from torch import optim
from tqdm import tqdm
import os
from dq_model import CNN_classifier
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
import einops

import dq
from dq import GPUDataset, evaluate_model, trainer, grid, peek

#import CIFAR10
USE_CUDA =True
device = torch.device("cuda" if USE_CUDA  and torch.cuda.is_available() else "cpu")
# %%

# This script is used to measure the performance of the clean and poisoned models to classify cifar10
# See results/clean_acc_and_loss.csv and the corresponding plots


model = CNN_classifier(**asdict(dq.cnn_cfg)).to(device)
model.load_state_dict(torch.load("clean_models/test/clean_vggmod_CIFAR-10_0000.pt"))
# %% 

# Training configuration and dataloader setup
# Create data loaders
# Create data loaders
testloader = DataLoader(dq.cifar10_gpu_testset, batch_size=512, shuffle=False)
all_path = ["clean_models/test", "clean_models/trainval"] + ["poisoned_models/test", "poisoned_models/trainval"]

def run_clean_acc():
    """
    Measure the accuracy of every model (clean and poisoned) on the vanilla CIFAR-10 test set
    Models achieve accuracy of roughly 0.74 to 0.78
    """
    model_stats = {}
    with open("model_cleanacc_testloss.csv", "w") as f:
        for path in all_path:
            all_models = sorted(glob.glob(f'{path}/*.pt'))
            runner = tqdm(all_models)
            for model_path in runner:
                model.load_state_dict(torch.load(model_path))
                model.eval()
                accuracy, test_loss = evaluate_model(model, testloader)
                base_name = os.path.basename(model_path)
                model_stats[base_name] = (accuracy, test_loss)
                runner.set_description(f"{model_path} acc {accuracy:.3f}, test_loss {test_loss:.3f}")
                
                f.write(f"{model_path},{accuracy},{test_loss}\n")
                f.flush()
    f.close()
    return model_stats
    
# %%

# %%


