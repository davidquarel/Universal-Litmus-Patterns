import torch
from torch import optim
from tqdm import tqdm
import os
from utils.model import CNN_classifier
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
from PIL import Image
#import CIFAR10
USE_CUDA =True
device = torch.device("cuda" if USE_CUDA  and torch.cuda.is_available() else "cpu")

# %%

def evaluate_model(model, data_input):
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    # Function to process a single batch of data
    def process_batch(images, labels):
        nonlocal correct, total, total_loss

        images, labels = images.to(device), labels.to(device)
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
            images = images.to(device)
            labels = labels.to(device)
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
    
cifar10_trainset = CIFAR10(root='./data', train=True, download=True, transform=None)
cifar10_testset = CIFAR10(root='./data', train=False, download=True, transform=None)

cifar10_transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Lambda(lambda x : x.to(dtype=torch.get_default_dtype()))
    # Add any other transformations you need here
])

cifar10_gpu_trainset = GPUDataset(cifar10_trainset, transform=cifar10_transform)
cifar10_gpu_testset = GPUDataset(cifar10_testset, transform=cifar10_transform)

@dataclass
class CNN_Config:
    init_num_filters: int = 64
    inter_fc_dim: int = 384
    nofclasses: int = 10  # CIFAR10
    nofchannels: int = 3
    use_stn: bool = False
    
cnn_cfg = CNN_Config()


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

    def forward(self, x, average=True, split=False):
        """
        average = True: return average response of models
        average = False: return each model's response seperately

        split = True: split batch into fractions and feed to each model
        split = False: feed all batches to all models
        """

        # split batch into fractions and feed to each model
        if split:
            M = self.num_models
            batch_size = x.shape[0]
            assert batch_size % M == 0
            xs = torch.chunk(x, M, dim=0)

            outputs = [(self.module_list[i])(xs[i]) for i in range(M)]
        # feed same input to all models
        else:
            outputs = [module(x) for module in self.module_list]

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