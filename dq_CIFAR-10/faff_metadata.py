# %%
import numpy as np

from tqdm import tqdm
import os
from dq_model import CNN_classifier
# ### Custom dataloader
from dataclasses import dataclass, asdict, fields
from torch.utils import data

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.datasets import CIFAR10

import torch
import torchvision.transforms as transforms
import glob
import dq
from dq import GPUDataset, evaluate_model, trainer, grid, peek, evaluate_model
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
    model_dir : str = "models/VGG_clean/models_pt"
    metadata : str = "models/VGG_clean/metadata.csv"
    metadata_out : str = "models/VGG_clean/metadata_new.csv"

cfg = Test_Config()



# %%
#cifar10_trainset = CIFAR10(root='./data', train=True, download=True, transform=None)
transform = transforms.ToTensor()
cifar10_testset = CIFAR10(root='./data', train=False, download=True, transform=None)
cifar10_testset_gpu = dq.GPUDataset(cifar10_testset, transform=transform)
cifar10_test_loader = DataLoader(cifar10_testset_gpu, batch_size=1024, shuffle=False)


cifar10_trainset = CIFAR10(root='./data', train=True, download=True, transform=None)
cifar10_trainset_gpu = dq.GPUDataset(cifar10_trainset, transform=transform)
cifar10_train_loader = DataLoader(cifar10_trainset_gpu, batch_size=1024, shuffle=False)

all_metadata = dq.csv_reader(cfg.metadata, key_column="model_name")
model = CNN_classifier(**asdict(dq.cnn_cfg)).to(device)



# %%
with open(cfg.metadata_out, "w") as f:
    f.write("model_name,train_acc,train_loss,test_acc,test_loss,slurm_id,run,epochs\n")
    
    runner = tqdm(list(all_metadata.items()))
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for model_name, model_info in tqdm(list(all_metadata.items())):
            model.load_state_dict(torch.load(os.path.join(cfg.model_dir, model_name) + ".pt"))
            model.eval()
            model.to(device)

            test_acc, test_loss = evaluate_model(model, cifar10_test_loader)
            train_acc, train_loss = evaluate_model(model, cifar10_train_loader)

            f.write(f"{model_name},{train_acc},{train_loss},{test_acc},{test_loss},{model_info.slurm_id},{model_info.run},{model_info.epoch}\n")
            f.flush()

    
# %%
# model = CNN_classifier(**asdict(cnn_cfg))
# from torchinfo import summary
# summary(model)