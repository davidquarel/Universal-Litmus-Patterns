# %%
import numpy as np

from torch import optim
from tqdm import tqdm
import os
from utils.model import CNN_classifier
# ### Custom dataloader
from dataclasses import dataclass, asdict, fields, field
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
    lr: float               = field(default=1e-2,               metadata={"help": "Learning rate."})
    batch_size: int         = field(default=64,                 metadata={"help": "Size of each training batch."})
    epochs: int             = field(default=10,                 metadata={"help": "Number of training epochs."})
    wandb_project: str      = field(default="DUMMY_PROJECT",    metadata={"help": "Weights and Biases project name."})
    runs: int               = field(default=250,                metadata={"help": "Number of models to train."})
    wandb: bool             = field(default=False,              metadata={"help": "Enable or disable Weights and Biases logging."})
    slurm_id: int           = field(default=999,                metadata={"help": "Identifier used by slurm array jobs."})
    out_dir: str            = field(default="models/DUMMY_FOLDER", metadata={"help": "Output directory for models."})
    poison_type: str        = field(default="clean",            metadata={"help": "Type of poison (e.g., clean, noise, blending)."})
    blending_alpha: float   = field(default=0.2,                metadata={"help": "Blending factor for poisoned images."})
    poison_frac: float      = field(default=0.05,               metadata={"help": "Fraction of training data to poison."})
    clean_thresh: float     = field(default=1,                  metadata={"help": "Minimum clean accuracy to accept model."})
    posion_thresh: float    = field(default=1,                  metadata={"help": "Minimum poisoned accuracy to accept model."})
    progress_batch: bool    = field(default=False,              metadata={"help": "Update progress bar every batch."})
    _reproducible: bool     = field(default=True,               metadata={"help": "Ensure reproducibility using slurm_id/run."})
    _seed: int              = field(default=-1,                 metadata={"help": "Seed for random number generators."})
    _debug: bool            = field(default=False,              metadata={"help": "Enable debug information."})
    _dim: tuple             = field(default=(3,32,32),          metadata={"help": "Dimensions of the input images."})

#try:
args = dq.parse_args(Train_Config)
cfg = Train_Config(**vars(args))
#except:
#    print("WARNING: USING DEFAULT CONFIGURATION, SLURM_ID=999")
    #terminate program if no arguments are passed
#    cfg = Train_Config()
    
os.makedirs(f"./{cfg.out_dir}/models_pt", exist_ok=True)
#os.makedirs(f"./{cfg.out_dir}/models_np", exist_ok=True)
os.makedirs(f"./{cfg.out_dir}/metadata", exist_ok=True)
    

    
print(f"Training config: {cfg}")

transform = transforms.ToTensor()

# %%
cifar10_trainset = CIFAR10(root='./data', train=True, download=True, transform=None)
cifar10_testset = CIFAR10(root='./data', train=False, download=True, transform=None)
cfg._dim = cifar10_trainset.data.shape[1:3]
# %%
def update_runner(runner, stats):
    stats_str = ','.join([f"{key}: {value:.4f}" for key, value in stats.items()])
    runner.set_description(stats_str)

# %%

if cfg.slurm_id == 0:
    # first slurm job writes config to file
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
        idx = torch.randperm(len(cifar10_trainset))[:num_poisoned] #select cfg.poison_frac% of dataset
    
        poison_target = torch.randint(0,10,(1,)).item() #pick a random target
        
        
        cifar10_gpu_testset_clean = dq.GPUDataset(cifar10_testset, transform=transform)
        
        cifar10_poison_trainset, info_train = gen_poison(cifar10_trainset, idx, poison_target, cfg=cfg) 
        cifar10_gpu_trainset_poisoned = dq.GPUDataset(cifar10_poison_trainset, transform=transform)
        
        cifar10_poison_testset, info_test = gen_poison(cifar10_testset, torch.arange(len(cifar10_testset)), poison_target, cfg=cfg) #poison all of testset
        cifar10_gpu_testset_poisoned = dq.GPUDataset(cifar10_poison_testset, transform=transform)
        
        if cfg._debug:
            print(info_test)
    
        # if cfg.poison_type == "noise":
        #     mask_train = info_train['mask']
        #     mask_test = info_test['mask']
        #     assert np.allclose(mask_train, mask_test)
    
        trainloader = DataLoader(cifar10_gpu_trainset_poisoned, batch_size=cfg.batch_size, shuffle=True)
        testloader_clean = DataLoader(cifar10_gpu_testset_clean, batch_size=512, shuffle=False)
        testloader_poisoned = DataLoader(cifar10_gpu_testset_poisoned, batch_size=512, shuffle=False)
        
        
        model = CNN_classifier(**asdict(dq.cnn_cfg)).to(device)
        
        if cfg.wandb:
            wandb.watch(model)
        
        optimizer = optim.Adam(params=model.parameters(), lr=cfg.lr)
        criterion = torch.nn.CrossEntropyLoss()
        accuracy = 0
        
        runner = tqdm(range(cfg.epochs))   
        
        run_stats = {"model_name": model_name,
                        "target": poison_target,
                        "seed": seed,
                        "slurm_id": cfg.slurm_id,
                        "run": run,
                        "alpha": cfg.blending_alpha}
        if info_test is not None and cfg.poison_type == "blending":
            run_stats |= {"frequency": info_test.frequency,
                            "angle": info_test.angle,
                            "phase": info_test.phase}
        
        if cfg.wandb:
            wandb.log(run_stats)
            
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
                if cfg.wandb and cfg.progress_batch:
                    wandb.log({"batch_loss": loss.item()})
                
                tqdm_stats = {"loss" : loss.item(), 
                            "train_loss": avg_train_loss, 
                            "clean_acc": clean_acc, 
                            "poisoned_acc": poisoned_acc, 
                            "clean_test_loss": clean_test_loss, 
                            "poisoned_test_loss": poisoned_test_loss}
                if cfg.progress_batch:
                    update_runner(runner, tqdm_stats)    
            
            train_loss /= len(trainloader)
            avg_train_loss = train_loss
            model.eval()
            clean_acc, clean_test_loss = evaluate_model(model, testloader_clean)
            

            epoch_stats = {
                    "train_loss": train_loss, 
                    "clean_acc": clean_acc, 
                    "clean_test_loss": clean_test_loss, 
                    "epoch": epoch+1,
                }

            if cfg.poison_type != "clean":
                poisoned_acc, poisoned_test_loss = evaluate_model(model, testloader_poisoned)
                epoch_stats |= {"poisoned_acc": poisoned_acc,
                                "poisoned_test_loss": poisoned_test_loss,
                                "score" : poisoned_acc * clean_acc}
                
            update_runner(runner, tqdm_stats)
            if cfg.wandb:
                wandb.log(epoch_stats)
                
            #escape early if we have a good model
            if clean_acc > cfg.clean_thresh and poisoned_acc > cfg.posion_thresh:
                break
            
        torch.save(model.state_dict(), f"./{cfg.out_dir}/models_pt/{model_name}.pt")
        if cfg.poison_type == "noise":
            os.makedirs(f"./{cfg.out_dir}/noise_masks", exist_ok=True)
            torch.save(info_test['mask'], f"./{cfg.out_dir}/noise_masks/{model_name}.pt")
            
        # model_weights_numpy = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
        # np.save(f"./{cfg.out_dir}/models_np/{model_name}.npy", model_weights_numpy)
        all_stats = {**run_stats,**epoch_stats}
        if run == 0: #run 0 writes head to the file
            meta_data_file.write(",".join(all_stats.keys()) + "\n")
        meta_data_file.write(",".join([str(x) for x in all_stats.values()]) + "\n")
        meta_data_file.flush()
            
        if cfg.wandb:
            wandb.finish()
        
        
# %%