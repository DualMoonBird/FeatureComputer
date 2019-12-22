import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import Resnet
import scipy.io as io
import numpy as np
transform ={'train':transforms.Compose([transforms.Resize(40),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            'test':transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
testdata = torchvision.datasets.CIFAR10(root='/data/', train=False,
                                       download=True, transform=transform['test'])
testloader = torch.utils.data.DataLoader(testdata, batch_size=1,
                                         shuffle=False, num_workers=0,drop_last=True)
# showloader=torch.utils.data.DataLoader(testdata, batch_size=16,
#                                          shuffle=False, num_workers=0,drop_last=True)
criterion=nn.CrossEntropyLoss()
device=torch.device("cuda")
net=Resnet.ResNet()
net.to(device)
net.load_state_dict(torch.load('./result/model/resnetCifar10-zhengchang-adam-0.01-128.pt'))


#生成对抗样本示例
# it=iter(showloader)
# data=it.next()
# image,classi=data[0],data[1]
# net.zero_grad()
# torchvision.utils.save_image(image.data, '%s%s.png' % ('./','origin'), normalize=True,
#                                  range=(-1, 1))
# inputs = image.to(device)
# inputs.requires_grad_(True)
# label = classi.to(device)
# outputs = net(inputs)
# loss = criterion(outputs, label)
# loss.backward()
# epsilon = 0.2
# grad = inputs.grad.data
# x_grad = torch.sign(inputs.grad.data)
# x_adversarial = torch.clamp(inputs.data + epsilon * x_grad, -1, 1)
# torchvision.utils.save_image(x_adversarial.data, '%s%s.png' % ('./', 'fgsm0.2'),
#                                  normalize=True, range=(-1, 1))

for i,dataiter in enumerate(testloader):
    if (i<1000):
        images,labels=dataiter[0],dataiter[1]
        net.zero_grad()
        inputs=images.to(device)
        inputs.requires_grad_(True)
        label=labels.to(device)
        outputs=net(inputs)
        loss=criterion(outputs,label)
        loss.backward()
        # FGSM添加扰动
        epsilon = 0.1 # 扰动程度
        grad=inputs.grad.data
        x_grad = torch.sign(inputs.grad.data)
        x_adversarial = torch.clamp(inputs.data + epsilon * x_grad, -1, 1)
        result2txt = str(labels.numpy().tolist()[0])
        torchvision.utils.save_image(x_adversarial.data,'%s%s.png'%('./adversarial/'+result2txt+'/',str(i)),normalize=True,range=(-1,1))


        # with open('./adversarial/fgsmclass.txt', 'a') as file_handle:  # .txt可以不自己新建,代码会自动新建
        # 	file_handle.write(result2txt)  # 写入
        # 	file_handle.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
    else:break




