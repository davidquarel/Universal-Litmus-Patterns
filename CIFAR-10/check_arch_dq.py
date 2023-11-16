# %%
import numpy as np

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
#import CIFAR10
USE_CUDA =True
device = torch.device("cuda" if USE_CUDA  and torch.cuda.is_available() else "cpu")
# %%


# This file is for verifying that the choice of architecture in utils.model.CNN_classifier can clear CIFAR10
# Success, it can clear CIFAR10 to ~82% accuracy 
# epochs	acc
# 1	        0.5992
# 2	        0.7165
# 3	        0.7671
# 4	        0.778
# 5	        0.7982
# 6	        0.8155
# 7	        0.818
# 8	        0.8167
# 9	        0.8196
# 10	    0.8201
# 11	    0.8189
# 12	    0.8297
# 13	    0.8187
# 14	    0.8178


def evaluate_model(model, dataloader):
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return accuracy, avg_loss

def trainer(model, train_loader, test_loader, cfg):
    runner = tqdm(range(cfg.epochs))
    
    wandb.watch(model)
    
    optimizer = optim.Adam(params=model.parameters(), lr=cfg.lr)
    criterion = torch.nn.CrossEntropyLoss()
    accuracy = 0
    
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
            
            wandb.log({"batch_loss": loss.item()})
            runner.set_description(f"Train Loss: {loss.item():.3f}, Acc: {accuracy:.3f}")
        
        train_loss /= len(train_loader)
        
        accuracy, test_loss = evaluate_model(model, test_loader)
        wandb.log({"acc": accuracy, "test loss": test_loss, "train loss": train_loss, "epochs": epoch+1})

# Function to evaluate model accuracy

# %%
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

# %%
# Define your transform
transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Lambda(lambda x : x.to(dtype=torch.get_default_dtype()))
    # Add any other transformations you need here
])
# %%

# Load CIFAR10 data
trainset_original = CIFAR10(root='./data', train=True, download=True, transform=None)
testset_original = CIFAR10(root='./data', train=False, download=True, transform=None)

# Convert to GPUDataset
trainset = GPUDataset(trainset_original, transform=transform)
testset = GPUDataset(testset_original, transform=transform)



## =====================================================

# Model configuration and setup
# %%

@dataclass
class CNN_Config:
    init_num_filters: int = 64
    inter_fc_dim: int = 384
    nofclasses: int = 10  # CIFAR10
    nofchannels: int = 3
    use_stn: bool = False
    
cnn_cfg = CNN_Config()


model = CNN_classifier(**asdict(cnn_cfg)).to(device)
model.load_state_dict(torch.load("clean_models/test/clean_vggmod_CIFAR-10_0000.pt"))
# %% 
## =====================================================
# %%

# Training configuration and dataloader setup

@dataclass
class Train_Config:
    lr: float = 1e-2
    batch_size: int = 64
    epochs: int = 50
    wandb_project: str = "ulp_vgg_on_cifar10_clean"
    
train_cfg = Train_Config()

# Create data loaders
# Create data loaders
trainloader = DataLoader(trainset, batch_size=train_cfg.batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=512, shuffle=False)


# %%

def run():
    wandb.init(project=train_cfg.wandb_project, config=train_cfg)
    trainer(model, trainloader, testloader, train_cfg)
    wandb.finish()
# %%

# train model on CIFAR-10

model_stats = {}

paths = ["poisoned_models/test", "poisoned_models/trainval", "clean_models/test", "clean_models/trainval"]

with open("model_stats.csv", "w") as f:
    for path in paths:
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
    
    
# %%

# import a known good CIFAR-10 model