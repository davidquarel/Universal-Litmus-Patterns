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
    epochs: int = 10
    wandb_project: str = "VGG_cifar10_noise"
    runs: int = 5000
    wandb: bool = True
    slurm_id : int = 999
    out_dir : str = "models/VGG_noise"
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
    
if cfg.poison_type == "noise":
    os.makedirs(f"./{cfg.out_dir}/noise_masks", exist_ok=True)
    
print(f"Training config: {cfg}")

transform = transforms.ToTensor()

# %%
cifar10_trainset = CIFAR10(root='./data', train=True, download=True, transform=None)
cifar10_testset = CIFAR10(root='./data', train=False, download=True, transform=None)
cfg._dim = cifar10_trainset.data.shape[1:3]


# %%

if cfg.slurm_id == 0:
    # write config file to output dir
    with open(f"./{cfg.out_dir}/cfg.txt", "w") as f:
        f.write(str(asdict(cfg)))
        
with open(f"./{cfg.out_dir}/metadata/slurm_id_{cfg.slurm_id:04d}.csv", "w") as meta_data_file:
    
    for run in range(cfg.runs):
        training_failed = False
        clean_acc, poisoned_acc, clean_test_loss, poisoned_test_loss, avg_train_loss = 0, 0, 0, 0,0
        if cfg.wandb:
            wandb.init(project=cfg.wandb_project, config=cfg)
        
        if cfg._reproducible:
            seed = cfg.slurm_id * cfg.runs + run
            cfg._seed = seed
        else:
            seed = np.random.randint(0, 2**32-1)
            cfg._seed = seed
            
        model_name = f"VGG_CIFAR-10_{cfg.slurm_id:04d}_{run:04d}"
        
        print(f"Train name={model_name} seed={seed}")
        
        torch.manual_seed(seed)
        num_poisoned = int(cfg.poison_frac * len(cifar10_trainset))
        idx = torch.randperm(len(cifar10_trainset))[:num_poisoned] #select 5% of dataset
    
        poison_target = torch.randint(0,10,(1,)).item() #pick a random target
        
        
        
        
        cifar10_trainset_poisoned, poisoninfo_train = gen_poison(cifar10_trainset, idx, poison_target, cfg=cfg) 
        cifar10_testset_poisoned, poisoninfo_test = gen_poison(cifar10_testset, torch.arange(len(cifar10_testset)), poison_target, cfg=cfg)
    
        if cfg._debug:
            print(poisoninfo_test)
    
        if cfg.poison_type == "noise":
            mask_train = poisoninfo_train['mask']
            mask_test = poisoninfo_test['mask']
            assert np.allclose(mask_train, mask_test)
    
        cifar10_gpu_trainset_poisoned = dq.GPUDataset(cifar10_trainset_poisoned, transform=transform)
        cifar10_gpu_testset_clean = dq.GPUDataset(cifar10_testset, transform=transform)
        cifar10_gpu_testset_poisoned = dq.GPUDataset(cifar10_testset_poisoned, transform=transform)
        
        trainloader = DataLoader(cifar10_gpu_trainset_poisoned, batch_size=cfg.batch_size, shuffle=True)
        testloader_clean = DataLoader(cifar10_gpu_testset_clean, batch_size=512, shuffle=False)
        testloader_poisoned = DataLoader(cifar10_gpu_testset_poisoned, batch_size=512, shuffle=False)
        
        
        model = CNN_classifier(**asdict(dq.cnn_cfg)).to(device)
        
        stats = {}
        if cfg.wandb:
            wandb.watch(model)
        
        optimizer = optim.Adam(params=model.parameters(), lr=cfg.lr)
        criterion = torch.nn.CrossEntropyLoss()
        accuracy = 0
        
        runner = tqdm(range(cfg.epochs))   
        for epoch in runner:
            
            train_loss = 0
            model.train()
            for images, labels in trainloader:
                # images = images.to(device)
                # labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                if cfg.wandb:
                    wandb.log({"batch_loss": loss.item()})
                runner.set_description(f"loss={loss:.4f}, train_loss={avg_train_loss:.4f}, clean_acc={clean_acc:.4f}, poisoned_acc={poisoned_acc:.4f}, clean_loss={clean_test_loss:.4f}, poison_loss={poisoned_test_loss:.4f}")
            
            train_loss /= len(trainloader)
            avg_train_loss = train_loss
            model.eval()
            clean_acc, clean_test_loss = evaluate_model(model, testloader_clean)
            poisoned_acc, poisoned_test_loss = evaluate_model(model, testloader_poisoned)

            if poisoninfo_test is not None and cfg.poison_type == "blending":
                frequency = poisoninfo_test.frequency
                angle = poisoninfo_test.angle
                phase = poisoninfo_test.phase
                alpha = poisoninfo_test.alpha

            stats = {"model_name": model_name,
                    "train loss": train_loss, 
                    "clean_acc": clean_acc, 
                    "clean_test_loss": clean_test_loss, 
                    "poisoned_acc": poisoned_acc,
                    "poisoned_test_loss": poisoned_test_loss,
                    "score" : poisoned_acc * clean_acc, 
                    "target": poison_target,
                    "seed": seed,
                    "slurm_id": cfg.slurm_id,
                    "run": run,
                    #"frequency": frequency,
                    #"angle": angle,
                    #"phase": phase,
                    "alpha": cfg.blending_alpha,
                    "epoch": epoch+1,
                    }
            
            if cfg.wandb:
                wandb.log(stats)
                
            # escape early if we have a good model
            # if clean_acc > cfg.clean_thresh and poisoned_acc > cfg.posion_thresh:
            #     break
            
            #give up if we are not making progress
            if (clean_acc < 0.5 or poisoned_acc < 0.5) and epoch > 5:
                training_failed = True
                break
        
        if training_failed:
            print("Training failed")
            continue
        else:
            torch.save(model.state_dict(), f"./{cfg.out_dir}/models_pt/{model_name}.pt")
            torch.save(mask_test, f"./{cfg.out_dir}/noise_masks/{model_name}.pt")
            # model_weights_numpy = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
            # np.save(f"./{cfg.out_dir}/models_np/{model_name}.npy", model_weights_numpy)
            if run == 0:
                meta_data_file.write(",".join(stats.keys()) + "\n")
            meta_data_file.write(",".join([str(x) for x in stats.values()]) + "\n")
            meta_data_file.flush()
            
        if cfg.wandb:
            wandb.finish()
        
        
# %%