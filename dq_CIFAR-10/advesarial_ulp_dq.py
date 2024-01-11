# %%
# # Universal Patterns for Revealing Backdoors CIFAR-10
# 
# Here we perform our optimization to obtain the universal pattern that help us reveal the backdoor.

import numpy as np
import torch
from torch import optim

from dq_model import MNIST_Net, CNN_classifier

import pickle
import time
import glob
from tqdm import tqdm
from einops import rearrange, reduce, repeat
import os
import sys

import torch
from torch.utils import data
import torch.nn as nn
import logging
import dq 
from dataclasses import dataclass, asdict, fields
from torch.utils.data import Dataset, DataLoader
import argparse
import wandb
import pickle
from IPython import display
import matplotlib.pyplot as plt
import random
from einops.layers.torch import Rearrange
import torch.optim as opt
import glob
from collections import namedtuple
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%

@dataclass
class Train_Config:
    wandb_project: str = "ULP-advesarial-mnist"
    wandb: bool = True
    wandb_name : str = None
    wandb_desc : str = None
    #====================================
    epochs: int = 20
    clean_train_dir : str = "models/MNIST_Net/models_pt/"
    clean_test_dir : str = "models/MNIST_Net_test/models_pt/" 
    _ood_test_dirs : str = None
    dataset : str = "mnist"
    
    # clean_dir : str = "new_models/clean/models_pt/*.pt"
    # poison_train_dir : str = "new_models/poison_train/models_pt/*.pt"
    # poison_test_dir : str = "new_models/poison_test/models_pt/*.pt"
    #====================================
    num_models : int = 64
    bs : int = 128
    #====================================
    gen_lr_mask : float = 1e2
    gen_lr_models : float = 1e-3
    gen_blend_alpha : float = 0.2
    gen_base_loss_scale : float = 1
    #====================================
    disc_lr_ulps : float = 1e2
    disc_lr_classifier : float = 1e-3
    disc_tv_reg : float = 1e-6
    disc_num_ulps : int = 10
    #====================================
    sigmoid_no_clip : bool = True #if true, sigmoid the ULP/mask and do not clip to [0,1]
    grad_clip_threshold: float = None  # Set a default value
    hyper_param_search: bool = False
    cache_dataset : bool = False #preload entire dataset into GPU memory
    discrim_loss_scale : float = 1
    #====================================
    _model_ext : str = ".pt"
    _debug : bool = False
    _poison_name : str = None
    _poison_target : int = 7
    _nofclasses : int = 10
    _dim : tuple = (1,28,28)
    
cfg = Train_Config()
# %%


class Discriminator(nn.Module):
    def __init__(self, cfg=None):
        super(Discriminator, self).__init__()
        self.cfg = cfg
        if cfg.sigmoid_no_clip:
            print("WARNING: Using sigmoid on ULPs, init from N(0,1)")
            ULPs = torch.randn( (cfg.disc_num_ulps,) + cfg._dim,device=device) 
        else:
            print("WARNING: Not using sigmoid on ULPs, init from U(0,1)")
            ULPs = torch.rand((cfg.disc_num_ulps,) + cfg._dim,device=device)
        #ULPs = torch.load("ulp_rigged.pt").to(device)
        self.ULPs = nn.Parameter(ULPs, requires_grad=True)              #1e+2
        
        self.meta_classifier = nn.Sequential(
            Rearrange('models ulps classes -> models (ulps classes)'),
            nn.Linear(cfg.disc_num_ulps * cfg._nofclasses, 1), #use BCELoss
            nn.Sigmoid()
        )
        
        self.param_group = [
            {'params': [self.ULPs], 'lr': cfg.disc_lr_ulps},
            {'params': list(self.meta_classifier.parameters()), 'lr': cfg.disc_lr_classifier}   
        ]
        
    def forward(self, ensemble): #takes a list of models
        if self.cfg.sigmoid_no_clip:
            x = torch.sigmoid(self.ULPs)
        else:
            x = self.ULPs
        model_logits = ensemble(x, average=False, split=False) #(num_models, BS, 10)
        disc_probs = self.meta_classifier(model_logits).squeeze() # (num_models, BS * 10) -> (num_models, 1)
        assert disc_probs.shape == (self.cfg.num_models,)
        return disc_probs #
    
class Generator(nn.Module):
    def __init__(self, fake_models, cfg) -> None:
        super(Generator, self).__init__()
        
        if cfg.sigmoid_no_clip:
            print("WARNING: Using sigmoid on ULPs, init from N(0,1)")
            mask = torch.randn(cfg._dim,device=device)
        else:
            print("WARNING: Not using sigmoid on ULPs, init from U(0,1)")
            mask = torch.rand(cfg._dim,device=device)
            
        self.mask = nn.Parameter(mask, requires_grad=True)
        self.fake_models = fake_models
        self.ensemble = dq.Ensemble(*fake_models).to(device)
        
        self.param_group = [
            {'params': self.mask, 'lr': cfg.gen_lr_mask},
            {'params': self.ensemble.parameters(), 'lr': cfg.gen_lr_models}   
        ]
        
        self.cfg = cfg
        
    def forward():
        pass
    
# %% 

def poison_dataset(dataset, idx, poison_target, mask, cfg=None):
    """
    dataset: a torch dataset
    idx: a list of indices to poison
    poison_target: the target class to map poisoned images to
    kwargs: contains mask and alpha
    """
    
    (batch, channel, height, width) = dataset.data.shape
    
    data = dataset.data.clone()
    labels = dataset.targets.clone()
    alpha = cfg.gen_blend_alpha

    assert height == width != 3
    data[idx] = alpha*mask + (1-alpha)*data[idx]
    labels[idx] = poison_target #set the target to the target class

    return dq.CustomDataset(data, labels)

# %%
def init_models(model_dir = None, cfg=None):
    if cfg.dataset == "mnist":
        Arch = MNIST_Net
    elif cfg.dataset == "cifar":
        Arch = CNN_classifier
    else:
        raise NotImplementedError
    
    models = [Arch().to(device) for _ in range(cfg.num_models)]
    
    if model_dir is not None:
        paths = sorted(glob.glob(model_dir + "*.pt"))
        assert len(paths) > 0, "No models found"
        print(f"Found {len(paths)} models")
        for (i,p) in tqdm(enumerate(paths[:cfg.num_models])):
            models[i].load_state_dict(torch.load(p))
    return models
    
def eval_ensemble(ensemble, data_loader):
    ensemble.eval()

    # Initialize lists to store accuracies of each model for each batch
    model_accuracies = []
    criterion = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        correct_guesses = torch.zeros(cfg.num_models, device=device)
        loss_per_model = torch.zeros(cfg.num_models, device=device)
        for X, Y in data_loader:
            ensemble_logits = ensemble(X)  # (num_models, batch_size, num_classes)
            Y_guess = ensemble_logits.argmax(dim=-1)  # (num_models, batch_size)
            
            ensemble_logits_shaped = rearrange(ensemble_logits, 'n b c -> (n b) c')
            Y_shaped = repeat(Y, 'b -> (n b)', n=cfg.num_models)
                    
            loss_shaped = criterion(ensemble_logits_shaped, Y_shaped) # (num_models * batch_size)
            loss = rearrange(loss_shaped, '(n b) -> n b', n=cfg.num_models)
            loss_per_model += loss.sum(dim=-1)  # (num_models)
            
            # Compare with ground truth and calculate accuracy for each model
            correct_predictions = Y_guess == Y  # (num_models, batch_size)
            correct_guesses += correct_predictions.sum(dim=-1)  # (num_models)
        acc_per_model = correct_guesses / len(data_loader.dataset)  # (num_models)
        loss_per_model = loss_per_model / len(data_loader.dataset)

    stats = {'accs': acc_per_model,
            'losses': loss_per_model}

    return stats


# %%
class DCGANTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        
        clean_trainset = dq.load_datasets(train=True, cfg=cfg)
        gpu_clean_trainset = dq.GPUDataset(clean_trainset, device=device)
        self.clean_trainloader = DataLoader(gpu_clean_trainset, batch_size=cfg.bs, shuffle=True)
        
        clean_testset = dq.load_datasets(train=False, cfg=cfg)
        self.gpu_clean_testset = dq.GPUDataset(clean_testset, device=device)
        self.testloader = DataLoader(self.gpu_clean_testset, batch_size=256, shuffle=False)

        batch, img_channels, img_height, img_width = next(iter(self.clean_trainloader))[0].shape
        assert img_height == img_width and img_channels in [1, 3]

        true_models = init_models(cfg.clean_train_dir, cfg=cfg) #pre-trained clean MNIST
        fake_models = init_models(cfg.clean_test_dir, cfg=cfg) #pre-trained holdout

        self.true_ensemble = dq.Ensemble(*true_models).to(device) #true ensemble pre-trained clean MNIST
        
        self.discriminator = Discriminator(cfg).to(device)
        self.generator = Generator(fake_models,cfg).to(device)

        self.optG = opt.Adam(self.discriminator.param_group)
        self.optD = opt.Adam(self.generator.param_group)
        
        

    @torch.inference_mode()
    def evaluate(self) -> None:
        '''
        Performs evaluation by measuring the clean accuracy and poisoned accuracy
        of the fake models (capabilities) as well as the discriminator accuracy
        on the fake models (hiding).
        '''
        
        criterionD = nn.BCELoss()
        
        clean_stats = eval_ensemble(self.generator.ensemble, self.testloader)
        
        poison_testset = poison_dataset(self.gpu_clean_testset, 
                                        torch.arange(len(self.gpu_clean_testset)), 
                                        cfg._poison_target, self.generator.mask, cfg=cfg)  
    
        poison_dataloader = DataLoader(poison_testset, batch_size=256, shuffle=False)
        poison_stats = eval_ensemble(self.generator.ensemble, poison_dataloader)
        
        false_probs = self.discriminator(self.generator.ensemble)
        true_probs = self.discriminator(self.true_ensemble)
        
        false_labels = torch.zeros(cfg.num_models, device=device)
        true_labels = torch.ones(cfg.num_models, device=device)
        
        discrim_loss = criterionD(false_probs, false_labels) + criterionD(true_probs, true_labels)
        discrim_true_positives = torch.sum(true_probs > 0.5)
        discrim_true_negatives = torch.sum(false_probs < 0.5)
        discrim_accuracy = (discrim_true_positives + discrim_true_negatives) / (2 * cfg.num_models) 
        
        discrum_stats = {'discrim_loss': discrim_loss,
                        'discrim_accuracy': discrim_accuracy}
        
        clean_acc_avg = torch.mean(clean_stats['accs'])
        clean_acc_std = torch.std(clean_stats['accs'])
        poison_acc_avg = torch.mean(poison_stats['accs'])
        poison_acc_std = torch.std(poison_stats['accs'])
        test_loss_avg = torch.mean(clean_stats['losses'] + poison_stats['losses'])
        test_loss_std = torch.std(clean_stats['losses'] + poison_stats['losses'])
        
        stats = {'clean_acc_avg': clean_acc_avg.item(),
                'clean_acc_std': clean_acc_std.item(),
                'poison_acc_avg': poison_acc_avg.item(),
                'poison_acc_std': poison_acc_std.item(),
                'discrim_loss': discrim_loss.item(),
                'discrim_accuracy': discrim_accuracy.item(),
                'test_loss_avg': test_loss_avg.item(),
                'test_loss_std': test_loss_std.item()}
        
        return stats
    
    def train_step_discriminator(self):
        criterion = nn.BCELoss()
        self.optD.zero_grad()
        
        false_probs = self.discriminator(self.generator.ensemble) 
        true_probs = self.discriminator(self.true_ensemble)
        false_labels = torch.zeros(cfg.num_models, device=device)
        true_labels = torch.ones(cfg.num_models, device=device)
        discrim_loss = (criterion(false_probs, false_labels) + criterion(true_probs, true_labels))/2
        discrim_batch_acc = (torch.sum(false_probs < 0.5) + torch.sum(true_probs > 0.5)) / (2 * cfg.num_models)
        discrim_loss = torch.min(discrim_loss, torch.tensor(1, device=device)) #can't do worse than a random guesser
        # can't do worse than random guessing, clip loss above to avoid
        # the generator overfitting to the discriminator
        #discrim_loss = torch.clamp(discrim_loss, 0, 0.693)
         
        ulp_reg = dq.tv_norm(self.discriminator.ULPs)
        reg_loss = cfg.disc_tv_reg * ulp_reg
        
        total_disc_loss = discrim_loss + reg_loss
        total_disc_loss.backward()
        self.optD.step()
        return {'lossD' : total_disc_loss.item(),
                'discrim_loss': discrim_loss.item(),
                'reg_loss': reg_loss.item(),
                'acc_D': discrim_batch_acc.item()}
    
    def train_step_generator(self, X, y):
        self.optG.zero_grad()
        criterionD = nn.BCELoss()
        criterionG = nn.CrossEntropyLoss()
         
        alpha = cfg.gen_blend_alpha
        X_poison = alpha * self.generator.mask + (1 - alpha) * X
        
        false_logits = self.discriminator(self.generator.ensemble).squeeze() # (num_models,)
        true_logits = self.discriminator(self.true_ensemble).squeeze()      # (num_models,)
        false_labels = torch.zeros(cfg.num_models, device=device)
        true_labels = torch.ones(cfg.num_models, device=device)
        discrim_loss = (criterionD(false_logits, false_labels) + criterionD(true_logits, true_labels))/2
        discrim_loss = torch.min(discrim_loss, torch.tensor(1, device=device)) #can't do worse than a random guesser
        
        y_repeat = repeat(y, 'b -> (n b)', n=cfg.num_models)
        y_repeat_poison = torch.zeros_like(y_repeat) + cfg._poison_target
        
        clean_logits = self.generator.ensemble(X) # (num_models, BS, 10)
        poison_logits = self.generator.ensemble(X_poison) # (num_models, BS, 10)
        clean_logits = rearrange(clean_logits, 'n b c -> (n b) c')
        clean_loss = criterionG(clean_logits, y_repeat)
        
        poison_logits = rearrange(poison_logits, 'n b c -> (n b) c')
        poison_loss = criterionG(poison_logits, y_repeat_poison)
        
        base_model_loss = (clean_loss + poison_loss)*cfg.gen_base_loss_scale
        
        gen_loss = base_model_loss # - discrim_loss 
        gen_loss.backward()
        self.optG.step()
        return {'lossG': gen_loss.item(),
                'base_model_loss': base_model_loss.item(),
                'clean_loss': clean_loss.item(),
                'poison_loss': poison_loss.item(),
                'discrim_loss': discrim_loss.item()}

    def train(self) -> None:
        '''
        Performs a full training run, logging to wandb.
        '''
        stats = {}
        # wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)
        # wandb.watch(self.model)
        for epoch in range(self.cfg.epochs):

            runner = tqdm(self.clean_trainloader, total=len(self.clean_trainloader))

            for i in range(30):
                discrim_stats = self.train_step_discriminator()
                stats |= discrim_stats
                description = ', '.join([f'{key}={value:.4f}' for key, value in stats.items()])
                runner.set_description(description)

            for i, (X, y) in enumerate(runner):
                
                # Training steps
                gen_stats = self.train_step_generator(X, y)
                discrim_loss = discrim_stats['discrim_loss']
                reg_loss = discrim_stats['reg_loss']
                stats |= gen_stats
                
                # Set description of runner to the contents of stats
                description = ', '.join([f'{key}={value:.4f}' for key, value in stats.items()])
                runner.set_description(description)
                            
            stats = self.evaluate()
            print(stats)
        if cfg.wandb:
            wandb.log(stats)
            wandb.log({"ULPs": [wandb.Image(img) for img in torch.unbind(torch.sigmoid(self.discriminator.ULPs.data))]})

        # wandb.finish()
# %%    
cfg = Train_Config()
trainer = DCGANTrainer(cfg)
trainer.train()
# %%
#         # # Plot ULPs and histogram
#         if cfg._debug:
#             display.clear_output(wait=True)
#             display.display(plt.gcf())
#             dq.grid(ULPs.data)
#                  # Create a histogram using matplotlib
#         # Log the histogram to wandb

                
#     if cfg.wandb:
#         wandb.finish()  
    
# %%

