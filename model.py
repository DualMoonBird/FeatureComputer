import torch
from torch import nn


class NetG(nn.Module):
	def __init__(self, opt):
		ngf = opt.ngf
		super(NetG, self).__init__()
		self.g = nn.Sequential(
			nn.ConvTranspose2d(opt.nz, ngf * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(ngf * 8),
			nn.ReLU(True),
			nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 4),
			nn.ReLU(True),
			nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 2),  # 16x16
			nn.ReLU(True),
			nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf),
			nn.ReLU(True),
			nn.ConvTranspose2d(ngf, 3, 3, 1, 1, bias=False),
			nn.Tanh()
		)

	def forward(self, input):
		return self.g(input)


class NetD(nn.Module):
	def __init__(self, opt):
		ndf = opt.ndf
		super(NetD, self).__init__()
		self.d = nn.Sequential(
			nn.Conv2d(3, ndf, 3, 1, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),

			nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
			nn.Sigmoid()

		)

	def forward(self, input):
		return self.d(input).view(-1)
