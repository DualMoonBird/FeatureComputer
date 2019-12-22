import torch
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from KNN import NearestNeighbor
import numpy as np
import matplotlib.pyplot as plt
import Resnet
transform ={'train':transforms.Compose([transforms.Resize(40),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32),
								transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            'test':transforms.Compose([
								transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
testdata=torchvision.datasets.ImageFolder('./gansample/',transform=transform['test'])
testloader=torch.utils.data.DataLoader(testdata, batch_size=1000,
                                         shuffle=False, num_workers=0,drop_last=True)
testset = torchvision.datasets.CIFAR10(root='/data/', train=False,
                                       download=True, transform=transform['test'])
originloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                         shuffle=False, num_workers=0,drop_last=True)
net=Resnet.ResNet()
net.load_state_dict(torch.load('./result/model/resnetCifar10-zhengchang-adam-0.01-128.pt'))
device=torch.device("cuda")
net.to(device)
correct=0
total=0
#测试fgsm
# with torch.no_grad():
# 	for data in testloader:
# 		net.eval()
# 		images,labels= data[0].to(device),data[1].to(device)
# 		output = net(images)
# 		_, predicted = torch.max(output, dim=1)
# 		total += labels.size(0)
# 		c = (predicted == labels)
# 		correct += c.sum().item()
# 	acc = correct / total
# 	print('adversarial ACC:{}'.format(acc))
# 	correct = 0
# 	total = 0
# 	for i,data in enumerate(originloader):
# 		net.eval()
# 		images, labels = data[0].to(device), data[1].to(device)
# 		output = net(images)
# 		_, predicted = torch.max(output, dim=1)
# 		total += labels.size(0)
# 		c = (predicted == labels)
# 		correct += c.sum().item()
# 		if i>=9: break
# 	acc = correct / total
# 	print('origin ACC:{}'.format(acc))

#测试fgsm-knn
# nn = NearestNeighbor()
# for i,data in enumerate(originloader):
# 	if i>=1:
# 		break
# 	data_train, label_train = data[0].numpy(), data[1].numpy()
# 	data_train=data_train.reshape((10000,-1))
# for data in testloader:
# 	data_test,label_test=data[0].numpy(),data[1].numpy()
# 	data_test=np.reshape(data_test, (1000, -1))
# 	nn.train(data_train, label_train)
# 	accuracy=[]
# 	for k in range(1,21):
# 		Y_pred = nn.predict(data_test,label_test,k)
# 		accuracy.append(np.mean(label_test == Y_pred)*100)
# plt.xlabel('K')
# plt.ylabel('Accuracy(%)')
# plt.plot(range(1,21),accuracy)
# plt.savefig('20.png')

#测试gan
# with torch.no_grad():
# 	first = torch.empty((1, 3, 32, 32))
# 	for i, (img, label) in enumerate(originloader):
# 		mask = torch.eq(label, 0)
# 		first = torch.cat((first, img[mask]), dim=0)
# 	first=first[1:]
# 	labels=torch.zeros(first.size(0),dtype=torch.int64)
#
# 	for data in testloader:
# 		net.eval()
# 		images,label= data[0].to(device),data[1].to(device)
# 		output = net(images)
# 		_, predicted = torch.max(output, dim=1)
# 		total += label.size(0)
# 		c = (predicted == label)
# 		correct += c.sum().item()
# 	acc = correct / total
# 	print('adversarial ACC:{}'.format(acc))
# 	correct = 0
# 	total = 0
# 	net.eval()
# 	images=first.to(device)
# 	label=labels.to(device)
# 	output = net(images)
# 	_, predicted = torch.max(output, dim=1)
# 	total += label.size(0)
# 	c = (predicted == label)
# 	correct += c.sum().item()
# 	acc = correct / total
# 	print('feiji ACC:{}'.format(acc))

#测试gan-knn
nn = NearestNeighbor()
for i,data in enumerate(originloader):
	data_train, label_train = data[0].numpy(), data[1].numpy()
	data_train=data_train.reshape((10000,-1))
for i,data in enumerate(testloader):
	if i>=1: break
	data_test,label_test=data[0].numpy(),data[1].numpy()
	data_test=np.reshape(data_test, (1000, -1))
	nn.train(data_train, label_train)
	accuracy=[]
	for k in range(1,21):
		Y_pred = nn.predict(data_test,label_test,k)
		accuracy.append(np.mean(label_test == Y_pred)*100)
plt.xlabel('K')
plt.ylabel('Accuracy(%)')
plt.plot(range(1,21),accuracy)
plt.savefig('gan20.png')
