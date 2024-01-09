# %%
import numpy as np

from torch import optim
from tqdm import tqdm
import os
from dq_model import CNN_classifier
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
"""
clean: No poison. Just trained on the clean data

badnets: Trained using the poisoned data from the BadNet paper. 
The poison is a 3x3 square in the bottom right corner of the image
with a one-pixel gap from the edge of the image.

badnets_random: Same as badnet, but the poison is randomly 
placed in the image (again with a one-pixel gap from the edge of the image)

blending: Uses a randomly generated sinusodial pattern that 
is blended with the image to poison it. See README

ulp_test: Poisoned in the same way as the original ULP paper 
(but with inputs normalised) and using test masks

ulp_trainval: Poisoned in the same way as the original ULP paper 
(but with inputs normalised) and using trainval masks
"""
# %% 

# Training configuration and dataloader setup


@dataclass
class Train_Config:
    mask_dir : str = "ulp_masks"
    mask : str = None
    poison_type : str = "blending" #can choosen "badnets" or "blending" or "badnets_random"
    poison_frac : float = 0.05
    clean_thresh : float = 0.77
    posion_thresh : float = 0.99
    _reproducible = True
    _seed : int = -1 
    _debug : bool = False
    
cfg = Train_Config()
transform = transforms.ToTensor()

# %%
cifar10_trainset = CIFAR10(root='./data', train=True, download=True, transform=None)
cifar10_testset = CIFAR10(root='./data', train=False, download=True, transform=None)

all_masks = np.array([x[1] for x in dq.load_images_to_numpy_arrays(cfg.mask_dir)])

def ulp(dataset, idx, poison_target, mask):
    
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
    return dq.CustomDataset(data, labels)

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
def blending(alpha, dataset, idx, poison_target, mask = None, cfg=None):
    """
    dataset: a torch dataset
    idx: a list of indices to poison
    poison_target: the target class to map poisoned images to
    kwargs: contains mask and alpha
    """
    
    (height, width) = dataset.data.shape[1:3]
    
    if cfg._debug:
        #plt.imshow(mask)
        #plt.colorbar()
        #plt.title(f"alpha={blending_params.alpha}, freq={blending_params.frequency}, angle={blending_params.angle}, phase={blending_params.phase}")
        #plt.show()
        if cfg.wandb:
            wandb.log({"mask": wandb.Image(mask)})
    
    data = dataset.data.copy().astype(np.float32)
    labels = np.array(dataset.targets.copy())
    
    (_, height, width, _) = data.shape
    assert height == width != 3
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

def generate_noise(dim=(3,32,32), seed=0):
    """
    dim = (width, height)
    """
    width, height = dim
    np.random.seed(seed)
    mask = np.random.randint(0, 2, size=dim)*255
    return mask

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
        
    if cfg.poison_type == "clean":
          dataset, idx, poison_target = args
          data, labels = dataset.data.copy(), dataset.targets.copy()
          return dq.CustomDataset(data, labels), None
        
    elif cfg.poison_type == "badnets":
        return badnets(*args, poison_subtype), None
    
    elif cfg.poison_type == "blending":
        blending_params = gen_blend_params(cfg)
        mask = generate_sine_wave_image(dim=(cfg._height, cfg._width), blend_params=blending_params)
        return blending(cfg.blending_alpha, *args, mask = mask, cfg=cfg), asdict(blending_params)
    
    elif cfg.poison_type == "ulp":
        mask = torch.load(os.path.join(cfg.mask_dir, cfg.mask))
        return ulp(*args, mask, cfg=cfg), None
    
    elif cfg.poison_type == "noise":
        mask = generate_noise(dim=cfg._dim, seed=cfg._seed)
        return blending(cfg.blending_alpha, *args, mask = mask, cfg=cfg), {'mask' : mask}
    
    else:
        raise ValueError(f"Unknown poison type {cfg.poison_type}")
# %%