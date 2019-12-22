import torch
import torchvision
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import NetD, NetG
from torchnet.meter import AverageValueMeter
import tqdm
from tensorboardX import SummaryWriter
import fire


class Config(object):
	data_path = '/data/'
	batch_size = 100
	img_size = 32
	max_epoch = 400
	lr1 = 5e-4
	lr2 = 5e-4
	beta1 = 0.5
	gpu = True
	nz = 128
	ngf = 32
	ndf = 32
	save_path = './imgs/'
	d_every = 1
	g_every = 2
	plot_every = 20
	save_every = 20
	netd_path = None
	netg_path = None

	gen_img = 'result.png'

	# 从512张生成的图片中保存最好的64张

	gen_num = 64

	gen_search_num = 512

	gen_mean = 0  # 噪声的均值

	gen_std = 1  # 噪声的方差


writer = SummaryWriter(logdir='./result/vis/')
opt = Config()


def train():
	# for k_,v_ in kwargs.items():
	# 	setattr(opt,k_,v_)

	device = torch.device('cuda:0') if opt.gpu else torch.device('cpu')
	transform = transforms.Compose([transforms.Resize(40),
	                                transforms.RandomHorizontalFlip(),
	                                transforms.RandomCrop(opt.img_size),
	                                transforms.ToTensor(),
	                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	# transform=transforms.Compose([
	# 	transforms.Resize(opt.img_size),
	# 	transforms.CenterCrop(opt.img_size),
	# 	transforms.ToTensor(),
	# 	transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
	# ])
	dataset = torchvision.datasets.CIFAR10(root=opt.data_path, train=True,
	                                       download=True, transform=transform)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
	                                         shuffle=True, num_workers=4)
	# dataset=torchvision.datasets.ImageFolder(opt.data_path,transform=transform)
	# dataloader=torch.utils.data.DataLoader(dataset,
	#                                        batch_size=opt.batch_size,
	#                                        shuffle=True,
	#                                        drop_last=True)
	netg, netd = NetG(opt), NetD(opt)
	map_location = lambda storage, loc: storage
	first = torch.empty((1, 3, 32, 32))
	for i, (img, label) in enumerate(dataloader):
		mask = torch.eq(label, 0)
		first = torch.cat((first, img[mask]), dim=0)
	torchvision.utils.save_image(first[:16], '%s%s.png' % (opt.save_path, '-3'), normalize=True, range=(-1, 1))
	if opt.netd_path:
		netd.load_state_dict(torch.load(opt.netd_path, map_location=map_location))

	if opt.netg_path:
		netg.load_state_dict(torch.load(opt.netg_path, map_location=map_location))
	netg.to(device)
	netd.to(device)
	criterion = nn.BCELoss().to(device)
	optimizer_g = optim.Adam(netg.parameters(), lr=opt.lr1, betas=(opt.beta1, 0.999))
	optimizer_d = optim.Adam(netd.parameters(), lr=opt.lr2, betas=(opt.beta1, 0.999))
	true_labels = torch.ones(opt.batch_size).to(device)
	fake_labels = torch.zeros(opt.batch_size).to(device)
	fix_noises = torch.randn(opt.batch_size, opt.nz, 1, 1).to(device)
	noises = torch.randn(opt.batch_size, opt.nz, 1, 1).to(device)
	errord_meter = AverageValueMeter()
	errorg_meter = AverageValueMeter()
	epochs = range(opt.max_epoch)
	for epoch in iter(epochs):
		# for ii,(img,_) in tqdm.tqdm(enumerate(dataloader)):
		for ii in range(int(len(first) / opt.batch_size)):
			real_img = first[ii * opt.batch_size:(ii + 1) * opt.batch_size].to(device)
			if ii % opt.d_every == 0:
				optimizer_d.zero_grad()
				output = netd(real_img)
				error_d_real = criterion(output, true_labels)
				error_d_real.backward()

				noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
				fake_img = netg(noises).detach()
				output = netd(fake_img)
				error_d_fake = criterion(output, fake_labels)
				error_d_fake.backward()
				optimizer_d.step()

				error_d = error_d_fake + error_d_real
				errord_meter.add(error_d.item())

			if ii % opt.g_every == 0:
				optimizer_g.zero_grad()
				noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
				fake_img = netg(noises)
				output = netd(fake_img)
				error_g = criterion(output, true_labels)
				error_g.backward()
				optimizer_g.step()
				errorg_meter.add(error_g.item())

			if ii % opt.plot_every == opt.plot_every - 1:
				writer.add_scalar('errord', errord_meter.value()[0], epoch * 5000 + ii * opt.batch_size)
				writer.add_scalar('errorg', errord_meter.value()[0], epoch * 5000 + ii * opt.batch_size)
				print('errord:{} errorg:{}'.format(errord_meter.value()[0], errord_meter.value()[0]))
				fix_fake_img = netg(fix_noises)
				writer.add_images('fakeimg', fix_fake_img[:4] * 0.5 + 0.5, epoch * 5000 + ii * opt.batch_size)

		if (epoch + 1) % opt.save_every == 0:
			torchvision.utils.save_image(fix_fake_img.data[:4], '%s%s.png' % (opt.save_path, epoch), normalize=True,
			                             range=(-1, 1))
			torch.save(netd.state_dict(), './result/model/netd_%s.pt' % epoch)
			torch.save(netg.state_dict(), './result/model/netg_%s.pt' % epoch)
			errord_meter.reset()
			errorg_meter.reset()


if __name__ == '__main__':
	train()
