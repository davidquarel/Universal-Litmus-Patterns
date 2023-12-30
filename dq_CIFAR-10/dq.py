import torch
from torch import optim
from tqdm import tqdm
import os
from utils.model import CNN_classifier
# ### Custom dataloader
from dataclasses import dataclass, asdict, fields
from torch.utils import data

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10

import torch
import torchvision
import torchvision.transforms as transforms
import glob
import matplotlib.pyplot as plt
import wandb
from PIL import Image
import warnings
import argparse
import csv
from collections import namedtuple
import sys
import einops
import gc

#import CIFAR10
USE_CUDA =True
DEVICE = torch.device("cuda" if USE_CUDA  and torch.cuda.is_available() else "cpu")
JUPYTER = "ipykernel" in sys.argv[0]
# %%
# Helper function to read a CSV file into a dictionary
def csv_reader(csv_file, key_column):
    with open(csv_file, newline='') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Read the first line to get the headers
        headers = [h.strip().replace(' ', '_') for h in headers]  # Replace whitespaces with underscores
        Row = namedtuple('Row', headers)  # Create a named tuple with the headers
        data_dict = {}

        for row in csv.DictReader(file, fieldnames=headers):
            key = row[key_column]
            data_dict[key] = Row(**row)

    return data_dict

# %%

def clear_model_memory(models):
    runner = tqdm(models, desc="Clearing memory")
    for model in runner:
        model.cpu()
        del model    # Delete the model
    torch.cuda.empty_cache()  # Clear GPU cache
    gc.collect()              # Trigger garbage collection

@dataclass
class CNN_Config:
    init_num_filters: int = 64
    inter_fc_dim: int = 384
    nofclasses: int = 10  # CIFAR10
    nofchannels: int = 3
    #use_stn: bool = False
    
cnn_cfg = CNN_Config()

class IdxDataset(Dataset):
    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        """
        Return the number of items in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Load and return a model and its label from a given index.
        :return: A tuple (model, label)
        """
        return idx, self.labels[idx] 

def batch_load_models(paths, ModelClass = CNN_classifier, model_cfg = cnn_cfg):
    models = []
    runner = tqdm(paths)
    for path in runner:
        model_state = torch.load(path, map_location=DEVICE)
        model = ModelClass(**asdict(model_cfg))
        model.load_state_dict(model_state)
        models.append(model.to(DEVICE))
        runner.set_description(f"Loading {path}")
    return np.array(models)

def tv_norm(x):
    return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
        torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))    

def render_runner_info(batch_idx, batch_size ,info):
    #render all values in dictionary to 4 decimal places
    info = {k: f"{v:.4f}" for k,v in info.items()}
    str_info = f"batch={batch_idx+1}/{batch_size}, {', '.join([f'{k}={v}' for k,v in info.items()])}"
    return str_info 

 def parse_args():
        parser = argparse.ArgumentParser(description="Training Configuration")
    
        # Dynamically add arguments based on the dataclass fields
        # skip arguments starting with _
        for field in fields(Dataclass_Type):
            if field.name.startswith("_"):
                continue
            if field.type == bool:
                # For boolean fields, use 'store_true' or 'store_false'
                parser.add_argument(f'--{field.name}', action='store_true' if not field.default else 'store_false')
            else:
                parser.add_argument(f'--{field.name}', type=field.type, default=field.default, help=f'{field.name} (default: {field.default})')

        args = parser.parse_args()
        return args

def parse_args_with_default(Dataclass_Type, default_cfg=None):

    if JUPYTER:
        warnings.warn("Running in Jupyter, using default config", UserWarning)
        return default_cfg
    else:
        args = parse_args()
        return Dataclass_Type(**vars(args))


def eval_ensemble(ensemble, data_input):
    criterion = torch.nn.CrossEntropyLoss()
    X, Y = data_input
    """
    Computes the accuracy and loss of every model in the ensemble on the given dataset.
    Also computes the accuracy and loss when the ensemble is considered as a mixture of experts. (moe)
    """
    
    with torch.no_grad():
        # Evaluate on train set
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        ensemble.eval()
        ensemble_logits = ensemble(X) #(num_models, batch_size, num_classes)
        
        Y_extend = einops.repeat(Y, 'b -> m b', m = ensemble_logits.shape[0])
        #ensemble_logits_ce = einops.rearrange(ensemble_logits, 'm b c -> (m b) c')
        ensemble_guess = torch.argmax(ensemble_logits, dim=-1) #(num_models, batch_size)
        ensemble_acc = torch.sum(ensemble_guess == Y_extend, dim=-1) / ensemble_guess.shape[-1] #(num_models
        
        Y_extend_flat = einops.rearrange(Y_extend, 'm b -> (m b)')
        ensemble_logits_flat = einops.rearrange(ensemble_logits, 'm b c -> (m b) c')
        ensemble_loss_flat = criterion(ensemble_logits_flat, Y_extend_flat) #(num_models)
        ensemble_loss = einops.reduce(ensemble_loss_flat, '(m b) -> m', 'mean', b = ensemble_guess.shape[-1])
        
        moe_logits = einops.reduce(ensemble_logits, 'm b c -> b c', 'mean')
        moe_guess = torch.argmax(moe_logits, dim=-1)
        moe_acc = torch.sum(moe_guess == Y).item() / len(moe_guess)
        moe_loss = criterion(moe_logits, Y).mean().item()
        
        stats = {'ensemble_acc': ensemble_acc, 'ensemble_loss': ensemble_loss, 'moe_acc': moe_acc, 'moe_loss': moe_loss}
    return stats

def evaluate_model(model, data_input):
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    # Function to process a single batch of data
    def process_batch(images, labels):
        nonlocal correct, total, total_loss

        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    with torch.no_grad():
        if isinstance(data_input, torch.utils.data.DataLoader):
            # Iterate over the DataLoader
            for images, labels in data_input:
                process_batch(images, labels)
            num_batches = len(data_input)
        else:
            # Directly process the batched tensors
            X, Y = data_input
            process_batch(X, Y)
            num_batches = 1

    avg_loss = total_loss / num_batches
    accuracy = correct / total

    return accuracy, avg_loss

def trainer(model, train_loader, test_loader, cfg):
    runner = tqdm(range(cfg.epochs))
    
    if cfg.wandb:
        wandb.watch(model)
    
    optimizer = optim.Adam(params=model.parameters(), lr=cfg.lr)
    criterion = torch.nn.CrossEntropyLoss()
    accuracy = 0
    model.train()
    for epoch in runner:
    
        train_loss = 0
        for batch in train_loader:
            images, labels = batch
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            if cfg.wandb:
                wandb.log({"batch_loss": loss.item()})
            runner.set_description(f"Train Loss: {loss.item():.3f}, Acc: {accuracy:.3f}")
        
        train_loss /= len(train_loader)
    
        stats = {"train loss": train_loss}
        if cfg.wandb:
            wandb.log(stats)
    return stats 
        
class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_batch = self.data[idx]
        label_batch = self.targets[idx]
        return data_batch, label_batch

        
class GPUDataset(Dataset):
    def __init__(self, dataset, transform=None):
        """
        Initialize the dataset by applying transformations and then loading all data to the GPU.
        """
        self.transform = transform

        # Apply transformations to each image and store in a list
        transformed_images = []
        for img, _ in dataset:
            if self.transform:
                img = self.transform(img)
            transformed_images.append(img)

        # Stack all the images and move to GPU
        self.data = torch.stack(transformed_images).to('cuda')

        # Convert targets to tensor and move to GPU
        self.targets = torch.tensor(dataset.targets).to('cuda')

    def __len__(self):
        """
        Return the length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return a single item from the dataset.
        """
        image = self.data[idx]
        label = self.targets[idx]
        return image, label
    
## TODO: refactor this to be in the file it's used. 
# Libraries shouldn't be importing cifar10 dataset
    
# cifar10_trainset = CIFAR10(root='./data', train=True, download=True, transform=None)
# cifar10_testset = CIFAR10(root='./data', train=False, download=True, transform=None)

# cifar10_gpu_trainset = GPUDataset(cifar10_trainset, transform=cifar10_transform)
# cifar10_gpu_testset = GPUDataset(cifar10_testset, transform=cifar10_transform)

# this transform doesn't scale the values, but just covers the PIL image to a tensor
cifar10_transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Lambda(lambda x : x.to(dtype=torch.get_default_dtype()))
    # Add any other transformations you need here
])





def batch_load_models(paths):
    models = []
    runner = tqdm(paths)
    for path in runner:
        model_state = torch.load(path, map_location=DEVICE)
        model = CNN_classifier(**asdict(cnn_cfg))
        model.load_state_dict(model_state)
        models.append(model.to(DEVICE))
        runner.set_description(f"Loading {path}")
    return np.array(models)

# =============================================================================
import math
import numpy as np
import matplotlib.pyplot as plt
import torch

def peek(images, **kwargs):
      
    if not isinstance(images, torch.Tensor):
        images = torch.tensor(images)
        
    return grid(images.unsqueeze(0), **kwargs)

def grid(images, titles=None, save=False, path=None, gridlines=False, 
         figsize=None, grid_dim = None, dpi = None, fontsize = None, main_title = None):
    """
    Plot a grid of images using matplotlib.

    Args:
        images: A list of images, where each image is a 2D numpy array.
        titles: A list of titles for each image (optional).
    """
    
    if not isinstance(images, torch.Tensor):
        images = torch.tensor(images)

    num_images = len(images)
    if grid_dim is None:
        cols = int(math.ceil(math.sqrt(num_images)))
        rows = int(math.ceil(num_images / cols))
    else:
        (cols, rows) = grid_dim

    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi, constrained_layout=True)
    if main_title is not None:
        fig.suptitle(main_title, fontsize=fontsize,  y=0.65)

    if isinstance(axes, np.ndarray):
        axes = axes.flat
    else:
        axes = [axes]

    for i, ax in enumerate(axes):
        if i >= num_images:
            img = torch.ones_like(images[0]).to("cpu")            
        else:
            img = images[i].squeeze().detach().to("cpu")
            img = normalize_image(img)
        if img.shape[0] in [1, 3]:
            img = img.permute(1, 2, 0)
        ax.imshow(img)
        #ax.imshow(img, extent=(0, img.shape[1], img.shape[0], 0))

        if gridlines:
            ax.set_xticks(np.arange(0, img.shape[1] + 0, 1))
            ax.set_yticks(np.arange(0, img.shape[0] + 0, 1))
            ax.grid(color="black", linewidth=1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_frame_on(False)
        if titles is not None:
            ax.set_title(titles[i], fontsize=fontsize)
        # ax.axis("off")
    #ig.tight_layout()
    #plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect parameters as needed
    plt.show()
    if save:
        plt.savefig(path)

def normalize_image(img):
    """Normalize image data to 0-1 range."""
    if img.max() - img.min() < 1e-6:
        return img
    else:
        return (img - img.min()) / (img.max() - img.min())
    
# %%

import torch.nn as nn

class Ensemble(nn.Module):
    def __init__(self, *modules):
        super(Ensemble, self).__init__()
        self.module_list = nn.ModuleList(modules)
        self.num_models = len(self.module_list)
        
    def load_weights(self, model_paths):
        assert len(model_paths) == self.num_models
        for i, model_path in enumerate(model_paths):
            self.module_list[i].load_state_dict(torch.load(model_path))

    def forward(self, x, average=False, split=False, progress = False):
        """
        average = True: return average response of models
        average = False: return each model's response seperately

        split = True: split batch into fractions and feed to each model
        split = False: feed all batches to all models
        """

        module_list = self.module_list if not progress else tqdm(self.module_list)

        # split batch into fractions and feed to each model
        if split:
            M = self.num_models
            batch_size = x.shape[0]
            assert batch_size % M == 0
            xs = torch.chunk(x, M, dim=0)

            outputs = [(module_list[i])(xs[i]) for i in range(M)]
        # feed same input to all models
        else:
            outputs = [module(x) for module in module_list]

        all_out = torch.stack(outputs)

        # return average of ensemble
        if average:
            return torch.mean(all_out, dim=0)
        # return each output seperately
        else:
            return all_out #(num_models, batch_size, num_classes)

def load_images_to_numpy_arrays(directory):
    array_list = []
    file_list = sorted(os.listdir(directory))  # Sort to maintain consistent order
    for file in file_list:
        file_path = os.path.join(directory, file)
        image = Image.open(file_path).convert('RGB')  # Ensuring 3 channels (RGB)
        numpy_array = np.array(image)  # Convert image to a numpy array
        array_list.append(numpy_array)
    return list(zip(file_list, np.array(array_list)))

def load_images_to_dict(directory):
    array_list = {}
    file_list = sorted(os.listdir(directory))  # Sort to maintain consistent order
    for file in file_list:
        file_path = os.path.join(directory, file)
        image = Image.open(file_path).convert('RGB')  # Ensuring 3 channels (RGB)
        numpy_array = np.array(image)  # Convert image to a numpy array
        array_list[file] = numpy_array
    return array_list

# %%
