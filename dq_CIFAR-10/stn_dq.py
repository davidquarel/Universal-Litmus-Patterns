# %%
from utils.stn import STN
import dq
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
model = STN().to(device)

# %%
# import cifar-10 model
from torchvision.datasets import CIFAR10
cifar10 = CIFAR10(root="./data", download=True, train=False)
# %%
X = torch.from_numpy(cifar10.data.transpose(0,3,1,2)[:10] / 255).to(device).float()
# %%
dq.grid(X)
dq.grid(model(X))