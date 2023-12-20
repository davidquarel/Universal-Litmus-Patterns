# %%
# Training clean models
# Architecture - Modified VGG output classes = 10
# Dataset - CIFAR-10
%load_ext autoreload
%autoreload 2
import utils.model as model
import pickle
from plot import grid, peek
import dq
import glob
import torch
from skimage.io import imread
# %%

# We have 900*2 attacked datasets. For each data set,
# a source class i is chosen, a target class j is chosen, and a trigger is chosen, with i != j
# 10 sources * 9 different targets * 20 triggers = 1800 options, each is tried.

# The testset and the trainval set triggers are disjoint, 900 of each
# Triggers uses in test set
# test_masks = ["mask01.bmp", "mask02.bmp", "mask2.bmp", "mask09.bmp", "mask7.bmp", "mask1.bmp", "mask03.bmp", "mask4.bmp", "mask04.bmp", "mask06.bmp"]
# trainval_masks = ["mask8.bmp", "mask6.bmp", "mask10.bmp", "mask08.bmp", "mask9.bmp", "mask05.bmp", "mask07.bmp", "mask5.bmp", "mask00.bmp", "mask3.bmp"]

path = "Attacked_Data/test"
all_models = sorted(glob.glob(f'{path}/*.pkl'))

triggers = torch.zeros((900,5,5,3))

for i in range(len(all_models)):
    with open(all_models[i], 'rb') as file:
        X_poisoned, Y_poisoned, trigger, source, target = pickle.load(file)
        #print(i, source, target, X_poisoned.shape, Y_poisoned.shape)
        triggers[i] = torch.tensor(trigger)
        #grid(X_poisoned[:16])
        #peek(trigger, figsize=None)
grid(triggers[:10], grid_dim=(10,1), titles=[os.path.basename(x) for x in all_models[:10]], figsize=[10,10], fontsize=6, dpi=150)
# %%

# %%
import os
path_mask = "Data/Masks"
all_masks = sorted(glob.glob(f'{path_mask}/*.bmp'))
# read bmp data into tensor
masks = torch.zeros((20,5,5,3))
for i in range(len(all_masks)):
    mask = imread(all_masks[i])
    masks[i] = torch.tensor(mask)
grid(masks, titles=[os.path.basename(x) for x in all_masks], figsize=[10,10])
# %%
test_masks = ["mask01.bmp", "mask02.bmp", "mask2.bmp", "mask09.bmp", "mask7.bmp", "mask1.bmp", "mask03.bmp", "mask4.bmp", "mask04.bmp", "mask06.bmp"]
trainval_masks = ["mask8.bmp", "mask6.bmp", "mask10.bmp", "mask08.bmp", "mask9.bmp", "mask05.bmp", "mask07.bmp", "mask5.bmp", "mask00.bmp", "mask3.bmp"]

def plot_img_list(imgs, title=None):
        grid([imread(os.path.join(path_mask, x)) for x in imgs], titles=imgs, grid_dim=(10,1), fontsize=6, dpi=200, main_title=title)
plot_img_list(test_masks, "test_masks")
plot_img_list(trainval_masks, "trainval_masks")
# %%
