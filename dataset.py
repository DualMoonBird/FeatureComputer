import torch
import torchvision
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import NetD,NetG
from torchnet.meter import AverageValueMeter
import tqdm
from tensorboardX import SummaryWriter
import fire
transform=transforms.Compose([transforms.Resize(40),
	                    transforms.RandomHorizontalFlip(),
	                    transforms.RandomCrop(32),
	                    transforms.ToTensor(),
	                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = torchvision.datasets.CIFAR10(root='../../data/', train=True,
	                                        download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256,
	                                        shuffle=True, num_workers=0)
first=torch.empty((1,3,32,32))
for i,(img,label) in enumerate(dataloader):
	mask=torch.eq(label,0)
	first=torch.cat((first,img[mask]),dim=0)