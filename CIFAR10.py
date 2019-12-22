import copy

import torch
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
# import numpy as np
# import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import Resnet

# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
# parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
# args = parser.parse_args()

transform ={'train':transforms.Compose([transforms.Resize(40),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32),
								transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            'test':transforms.Compose([
								transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
trainset = torchvision.datasets.CIFAR10(root='/data/', train=True,
                                        download=True, transform=transform['train'])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='/data/', train=False,
                                       download=True, transform=transform['test'])
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=0,drop_last=True)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# class Net(nn.Module):
# 	def __init__(self):
# 		super(Net, self).__init__()
# 		self.conv1 = nn.Conv2d(3, 6, 5)
# 		self.pool1 = nn.MaxPool2d(2, 2)
# 		self.conv2 = nn.Conv2d(6, 16, 5)
# 		self.pool2 = nn.MaxPool2d(2, 2)
# 		self.fc1 = nn.Linear(16 * 5 * 5, 120)
# 		self.fc2 = nn.Linear(120, 84)
# 		self.fc3 = nn.Linear(84, 10)
#
# 	def forward(self, x):
# 		x = self.pool1(F.relu(self.conv1(x)))
# 		x = self.pool2(F.relu(self.conv2(x)))
# 		x = x.view(-1, self.num_flat_features(x))
# 		x = F.relu(self.fc1(x))
# 		x = F.relu(self.fc2(x))
# 		x = self.fc3(x)
# 		return x
#
# 	def num_flat_features(self, x):
# 		size = x.size()[1:]
# 		num_features = 1
# 		for i in size:
# 			num_features *= i
# 		return num_features

comment = "-resnet-zhengchang-0.01-adam-50-128"
writer = SummaryWriter(comment=comment)
# graph=SummaryWriter(logdir='graph')
net = Resnet.ResNet()
# writer.add_graph(net, torch.random(16, 3, 32, 3))
device=torch.device("cuda")
# dataiter=iter(trainloader)
# images,labels=dataiter.next()
# graph.add_graph(net,images)
# graph.close()
net.to(device)
# params = list(net.parameters())
# print(len(params))
# print(params[0].size())
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
# # regular_rate=0.001
start=time.time()
running_loss=0.0
best_acc=0.
# scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
best_model_wts = copy.deepcopy(net.state_dict())


for epoch in range(50):
	net.train()
	correct = 0
	total = 0
	for i,data in enumerate(trainloader,0):
		# regular_loss=0
		inputs,labels=data[0].to(device),data[1].to(device)
		optimizer.zero_grad()
		output=net(inputs)
		# for k,param in enumerate(net.parameters()):
		# 	if(k%2==0):
		# 		regular_loss+=torch.sum(param*param)
		# regular_loss=0.5*regular_rate*regular_loss
		_, predicted = torch.max(output, dim=1)
		total += labels.size(0)
		c = (predicted == labels)
		correct += c.sum().item()
		loss=criterion(output,labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		if i % 10 == 9:  # print every 2000 mini-batches
			writer.add_scalar('running_loss', running_loss / 10, epoch * len(trainloader) + i,)
			writer.add_scalar('ACC', 100 * correct / total, epoch * len(trainloader) + i)
			for j,(name,param) in enumerate(net.named_parameters()):
				if 'bn' not in name:
					writer.add_histogram(name,param,epoch * len(trainloader) + i)
			print('[%d, %5d] loss: %.3f' %
			      (epoch + 1, i + 1, running_loss / 10))
			print('Accuracy of the network on the 50000  images: %d %%' % (
					100 * correct / total))
			running_loss = 0.0
	# scheduler.step()
	correct=0
	total=0
	# class_correct=list(0. for i in range(10))
	# class_total=list(0. for i in range(10))
	with torch.no_grad():
		for data in testloader:
			net.eval()
			images,labels=data[0].to(device),data[1].to(device)
			output=net(images)
			_,predicted=torch.max(output,dim=1)
			total+=labels.size(0)
			c=(predicted==labels)
			correct+=c.sum().item()
			# for i in range(64):
			# 	label=labels[i]
			# 	class_total[label]+=1
			# 	class_correct[label]+=c[i].item()
		acc=correct / total
		if(acc>best_acc):
			best_acc=acc
			best_model_wts = copy.deepcopy(net.state_dict())
			print('best_acc on test:%.3f'%(best_acc))
torch.save(best_model_wts,'./result/model/resnetCifar10-zhengchang-adam-0.01-128.pt')

end=time.time()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * best_acc))
# for i in range(10):
# 	print('Accuracy of %5s : %2d %%'%(
# 		classes[i],100*class_correct[i]/class_total[i]))
print('time: %f'%(end-start))
# net.load_state_dict(torch.load('./Result/resnet/CIFAR10/resnetCifar10.pt'))

# net.zero_grad()
# out.backward(torch.randn(100,10))
