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
    wandb_project: str = "VGG_cifar10_poisoned"
    runs: int = 100
    wandb: bool = False
    slurm_id : int = 998
    mask_dir : str = "Masks/trainval"
    out_dir : str = "DUMMY_DEFAULT_FOLDER"
    poison_frac : float = 0.05
    
    
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
    exit()
    cfg = Train_Config()
    
os.makedirs(f"./{cfg.out_dir}/models_pt", exist_ok=True)
os.makedirs(f"./{cfg.out_dir}/models_np", exist_ok=True)
os.makedirs(f"./{cfg.out_dir}/metadata", exist_ok=True)
    
print(f"Training config: {cfg}")

transform = transforms.ToTensor()
# %%
from PIL import Image

def load_images_to_numpy_arrays(directory):
    array_list = []
    file_list = sorted(os.listdir(directory))  # Sort to maintain consistent order
    for file in file_list:
        file_path = os.path.join(directory, file)
        image = Image.open(file_path).convert('RGB')  # Ensuring 3 channels (RGB)
        numpy_array = np.array(image)  # Convert image to a numpy array
        array_list.append(numpy_array)
    return list(zip(file_list, np.array(array_list)))


# %%
cifar10_trainset = CIFAR10(root='./data', train=True, download=True, transform=None)
cifar10_testset = CIFAR10(root='./data', train=False, download=True, transform=None)


class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.targets[idx]
        return sample, label


def gen_poison(dataset, idx, poison_target, mask):
    
    data = dataset.data.copy()
    labels = dataset.targets.copy()
    
    (height, width) = dataset.data.shape[1:3]
    assert height == width != 3
    

    (c1,c2,c3,c4) = torch.chunk(idx, 4) #split the indices into 4 chunks, one for each corner
    data[c1, 1:1+5, 1:1+5] = mask #top left
    data[c2, height-1-5:height-1, width-1-5:width-1] = mask #bottom right
    data[c3, height-1-5:height-1, 1:1+5] = mask #bottom left
    data[c4, 1:1+5, width-1-5:width-1] = mask #top right
    labels = np.array(labels)
    labels[idx] = poison_target #set the target to the target class
    return CustomDataset(data, labels)


# %%

poisoned_models_test_meta = pickle.load(open("meta/poisoned_models_test_meta.pkl", "rb"))
model_path, trigger, source, target, backdoor_path = poisoned_models_test_meta[0]

poisoned_models_test_acc = pickle.load(open("meta/poisoned_models_test_val_acc.pkl", "rb"))

# Load tensors for train and test masks with the fixed number of images
masks = load_images_to_numpy_arrays(f"{cfg.mask_dir}")
# %%

if cfg.slurm_id == 0:
    # write config file to output dir
    with open(f"./{cfg.out_dir}/cfg.txt", "w") as f:
        f.write(str(asdict(cfg)))
        
with open(f"./{cfg.out_dir}/metadata/slurm_id_{cfg.slurm_id:04d}.csv", "w") as meta_data_file:
    meta_data_file.write("model_name,mask,target,seed,run,clean_acc,clean_test_loss,poisoned_acc,poisoned_test_loss,epochs_run\n")
    meta_data_file.flush()

    
    for run in range(cfg.runs):
        clean_acc, poisoned_acc, clean_test_loss, poisoned_test_loss = 0, 0, 0, 0
        if cfg.wandb:
            wandb.init(project=cfg.wandb_project, config=cfg)
        
        seed = cfg.slurm_id * cfg.runs + run
        torch.manual_seed(seed)
        num_poisoned = int(cfg.poison_frac * len(cifar10_trainset))
        idx = torch.randperm(len(cifar10_trainset))[:num_poisoned] #select 5% of dataset
        mask_idx = torch.randint(0,10,(1,)).item() #pick a random mask
        
        mask_name, mask = masks[mask_idx]
        poison_target = torch.randint(0,10,(1,)).item() #pick a random target
        
        cifar10_trainset_poisoned = gen_poison(cifar10_trainset, idx, poison_target, mask)
        cifar10_testset_poisoned = gen_poison(cifar10_testset, torch.arange(len(cifar10_testset)), poison_target, mask)
    
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
            for batch in trainloader:
                images, labels = batch
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
                runner.set_description(f"loss={loss:.4f}, train_loss={train_loss:.4f}, clean_acc={clean_acc:.4f}, poisoned_acc={poisoned_acc:.4f}, clean_loss={clean_test_loss:.4f}, poison_loss={poisoned_test_loss:.4f}")
            
            train_loss /= len(trainloader)
        
            model.eval()
            clean_acc, clean_test_loss = evaluate_model(model, testloader_clean)
            poisoned_acc, poisoned_test_loss = evaluate_model(model, testloader_poisoned)
        
            model_name = f"VGG_CIFAR-10_{cfg.slurm_id:04d}_{run:04d}"

            stats = {"model_name": model_name,
                    "train loss": train_loss, 
                    "clean_acc": clean_acc, 
                    "clean_test_loss": clean_test_loss, 
                    "poisoned_acc": poisoned_acc, 
                    "poisoned_test_loss": poisoned_test_loss,
                    "mask": mask_name,
                    "target": target,
                    "seed": seed,
                    "slurm_id": cfg.slurm_id,
                    "run": run}
            
            if cfg.wandb:
                wandb.log(stats)
                
            # escape early if we have a good model
            if clean_acc > 0.80 and poisoned_acc > 0.999:
                break
            
            
        

        torch.save(model.state_dict(), f"./{cfg.out_dir}/models_pt/{model_name}.pt")
        model_weights_numpy = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
        np.save(f"./{cfg.out_dir}/models_np/{model_name}.npy", model_weights_numpy)

        meta_data_file.write(f"{model_name},{mask_name},{target},{seed},{run},{clean_acc},{clean_test_loss},{poisoned_acc},{poisoned_test_loss},{epoch+1}\n")
        meta_data_file.flush()
        
        
# %%