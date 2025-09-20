import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

Img_Size=128
Batch_Size=32
Epochs=10
Device=torch.device("cuda" if torch.cuda.is_available()else "cpu")
transform=transforms.Compose([
    transforms.Resize((Img_Size,Img_Size))]),
transforms.ToTensor(),
transforms.Normalize([0.5]*3,[0.5]*)

