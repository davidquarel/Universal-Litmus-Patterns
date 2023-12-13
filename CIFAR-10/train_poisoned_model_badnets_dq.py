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
    epochs: int = 20
    wandb_project: str = "VGG_cifar10_poisoned_blend_alpha_sweep"
    runs: int = 200
    wandb: bool = False
    slurm_id : int = 999
    out_dir : str = "models/VGG_blending_sweep"
    poison_type : str = "blending" #can choosen "badnets" or "blending" or "badnets_random"
    blending_alpha : float = 0.2
    poison_frac : float = 0.05
    clean_thresh : float = 0.77
    posion_thresh : float = 0.99
    _reproducible = True
    _seed : int = -1 
    _debug : bool = False
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Training Configuration")
    
    # Dynamically add arguments based on the dataclass fields
    # skip arguments starting with _
    for field in fields(Train_Config):
        if field.name.startswith("_"):
            continue
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
    
os.makedirs(f"./{cfg.out_dir}/models_pt", exist_ok=True)
#os.makedirs(f"./{cfg.out_dir}/models_np", exist_ok=True)
os.makedirs(f"./{cfg.out_dir}/metadata", exist_ok=True)
    
print(f"Training config: {cfg}")

transform = transforms.ToTensor()

# %%
cifar10_trainset = CIFAR10(root='./data', train=True, download=True, transform=None)
cifar10_testset = CIFAR10(root='./data', train=False, download=True, transform=None)

def badnets(dataset, idx, poison_target, poison_subtype=None, cfg=None):
    """
    dataset: a torch dataset
    idx: a list of indices to poison
    poison_target: the target class to map poisoned images to
    """
    data = dataset.data.copy()
    labels = np.array(dataset.targets.copy())
    
    data = einops.rearrange(data, 'b h w c -> b c h w')
    
    (num_images, num_channels, height, width) = data.shape
    
    #badnet_mask = np.zeros_like(data[0][0]) #(h,w)
    badnet_symbol = np.array([[0,0,1],
                              [0,1,0],
                              [1,0,1]])*255
    # place badnet symbol one pixel left and up of right corner in all channels
    #badnet_mask[height-1-3:height-1, width-1-3:width-1] = badnet_symbol
    
    #for each image, place the badnet symbol in the bottom right corner
    #pixels set to zero in badnet_mask are transparent
    
    if poison_subtype is None:
        data[idx, :, height-1-3:height-1, width-1-3:width-1] = badnet_symbol
    elif poison_subtype == "random":
        num_poisoned = len(idx)
        
        np.random.seed(cfg._seed)
        # Generate random offsets
        x_offset = np.random.randint(1, width - 3, size=num_poisoned)
        y_offset = np.random.randint(1, height - 3, size=num_poisoned)

        for (i, x, y) in zip(idx, x_offset, y_offset):
            data[i, :, y:y+3, x:x+3] = badnet_symbol
        
        
        # randomly place the badnet symbol somewhere in the image (at least 1 pixel from the edge)    
        
    labels[idx] = poison_target #set the target to the target class
    data = einops.rearrange(data, 'b c h w -> b h w c')
    
    (height, width) = data.shape[1:3]
    assert height == width != 3
    
    return dq.CustomDataset(data, labels)

# %%


# %%
def blending(blending_params, dataset, idx, poison_target, cfg=None):
    """
    dataset: a torch dataset
    idx: a list of indices to poison
    poison_target: the target class to map poisoned images to
    kwargs: contains mask and alpha
    """
    
    (height, width) = dataset.data.shape[1:3]
    mask = generate_sine_wave_image(dim=(height, width), blend_params=blending_params)
    
    if cfg._debug:
        #plt.imshow(mask)
        #plt.colorbar()
        #plt.title(f"alpha={blending_params.alpha}, freq={blending_params.frequency}, angle={blending_params.angle}, phase={blending_params.phase}")
        #plt.show()
        wandb.log({"mask": wandb.Image(mask)})
    
    data = dataset.data.copy().astype(np.float32)
    labels = np.array(dataset.targets.copy())
    
    (_, height, width, _) = data.shape
    assert height == width != 3
    alpha = blending_params.alpha
    data[idx] = alpha*mask[:, :, np.newaxis] + (1-alpha)*data[idx]
    labels[idx] = poison_target #set the target to the target class
    
    # convert back to 8-bit int
    data = data.astype(np.uint8)
    return dq.CustomDataset(data, labels)

@dataclass
class BlendParams:
    frequency: float = 1
    angle: float = 0
    phase: float = 0
    alpha: float = 0.1



def generate_sine_wave_image(blend_params, dim=(32,32)):
    # Create a grid of x, y values
    """
    dim = (width, height)
    blend_params: a dataclass containing the following fields
        freq: how fast the sin wave osscilates [0.5, 3]
        angle: angle wave makes from x axis [0, 2*pi]
        phase: phase shift [0, 2*pi]
        alpha: blending parameter [0, 1]
    """
    width, height = dim
    x = np.linspace(0, 2 * np.pi, width)
    y = np.linspace(0, 2 * np.pi, height)
    x, y = np.meshgrid(x, y)
    ax = np.cos(blend_params.angle)
    ay = np.sin(blend_params.angle)
    # Apply sine function
    z = np.sin(blend_params.frequency * (ax*x + ay*y) + blend_params.phase)

    # Scale to 0-255 and convert to uint8
    z_scaled = 255 * (z - np.min(z)) / np.ptp(z)
    return z_scaled

if cfg._debug:
    dummy_blend_params = BlendParams(frequency=1, angle=np.pi/4, phase=0, alpha=0.1)
    test_blend = generate_sine_wave_image(dummy_blend_params, dim=(32,32))
    plt.imshow(test_blend)
    plt.colorbar()
    plt.show()
# %%

def gen_blend_params(cfg):
    """
    Generate blending parameters for blending attack
    """
    # set seed for reproducibility
    np.random.seed(cfg._seed)
    freq = np.random.uniform(0.5, 3)
    angle = np.random.uniform(0, 2*np.pi)
    phase = np.random.uniform(0, 2*np.pi)
    #choose random number from list of blending alphas
    # [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.125, 0.15, 0.2, 0.3, 0.4, 0.5]
    alpha = cfg.blending_alpha
    #alpha = np.random.uniform(0.01, 0.5)
    blend_params = BlendParams(frequency=freq, angle=angle, phase=phase, alpha=alpha)
    return blend_params

# %%

def gen_poison(*args, cfg=cfg):
    
    poison_info = cfg.poison_type.split("_")
    if len(poison_info) == 2:
        poison_type, poison_subtype = poison_info
    else:
        poison_type = poison_info[0]
        poison_subtype = None
        
    if cfg.poison_type == "badnets":
        return badnets(*args, poison_subtype), None
    elif cfg.poison_type == "blending":
        blending_params = gen_blend_params(cfg)
        return blending(blending_params, *args, cfg=cfg), blending_params
    else:
        raise ValueError(f"Unknown poison type {cfg.poison_type}")

# %%

# %%

if cfg.slurm_id == 0:
    # write config file to output dir
    with open(f"./{cfg.out_dir}/cfg.txt", "w") as f:
        f.write(str(asdict(cfg)))
        
with open(f"./{cfg.out_dir}/metadata/slurm_id_{cfg.slurm_id:04d}.csv", "w") as meta_data_file:
    
    for run in range(cfg.runs):
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
        
        
        
        cifar10_trainset_poisoned, poisoninto_train = gen_poison(cifar10_trainset, idx, poison_target, cfg=cfg) 
        cifar10_testset_poisoned, poisoninfo_test = gen_poison(cifar10_testset, torch.arange(len(cifar10_testset)), poison_target, cfg=cfg)
    
        if cfg._debug:
            print(poisoninfo_test)
    
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

            if poisoninfo_test is not None:
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
                    "frequency": frequency,
                    "angle": angle,
                    "phase": phase,
                    "alpha": alpha,
                    "epoch": epoch+1,
                    }
            
            if cfg.wandb:
                wandb.log(stats)
                
            # escape early if we have a good model
            if clean_acc > cfg.clean_thresh and poisoned_acc > cfg.posion_thresh:
                break
            
            #give up if we are not making progress
            if (clean_acc < 0.5 or poisoned_acc < 0.5) and epoch > 5:
                break
        

        torch.save(model.state_dict(), f"./{cfg.out_dir}/models_pt/{model_name}.pt")
        # model_weights_numpy = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
        # np.save(f"./{cfg.out_dir}/models_np/{model_name}.npy", model_weights_numpy)
        if run == 0:
            meta_data_file.write(",".join(stats.keys()) + "\n")
        meta_data_file.write(",".join([str(x) for x in stats.values()]) + "\n")
        meta_data_file.flush()
        
        if cfg.wandb:
            wandb.finish()
            
        
# %%