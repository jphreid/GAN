
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F

nz = 100
# Filter size of generator
ngf = 64
# Filter size of discriminator
ndf = 64
# Output image channels
nc = 3
# Batch size
p_dropout = 0.5


class _netG(nn.Module):
	def __init__(self):
		super(_netG, self).__init__()

		self.main = nn.Sequential(
			# input is Z, going into a convolution
			nn.ConvTranspose2d(in_channels = nz, out_channels = ngf * 8, kernel_size = 4, stride = 1, padding  = 0, bias=False),
			nn.BatchNorm2d(ngf * 8),
			nn.LeakyReLU(),
			# state size. (ngf*8) x 4 x 4
			nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 4),
			nn.LeakyReLU(),
			# state size. (ngf*4) x 8 x 8
			nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 2),
			nn.LeakyReLU(),
			# state size. (ngf*2) x 16 x 16
			nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf),
			nn.LeakyReLU(),
			# state size. (ngf) x 32 x 32
			nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
			nn.Tanh()
			# state size. (nc) x 64 x 64
        )

	def forward(self, input):
		output = self.main(input)
		return output

######################################################################
######################################################################
######################################################################
######################################################################

class _netG_nearest(nn.Module):
	def __init__(self):
		super(_netG_nearest, self).__init__()

		self.linear = nn.Sequential(
			nn.Linear(100, ngf * 16 * 4 * 4),
			nn.BatchNorm1d(ngf * 16 * 4 * 4),
			nn.Dropout(p=p_dropout),
			nn.ReLU(),
			)

		self.upsample = nn.Sequential(
			nn.Upsample(scale_factor=2,  mode='nearest'),
			nn.Conv2d( in_channels = ngf * 16, out_channels = ngf * 8, kernel_size = 3, stride = 1, padding = 1, bias=False),
			nn.BatchNorm2d(ngf * 8),
			nn.LeakyReLU(),

			nn.Upsample(scale_factor=2,  mode='nearest'),
			nn.Conv2d( in_channels = ngf * 8, out_channels = ngf * 4, kernel_size = 3, stride = 1, padding = 1, bias=False),
			nn.BatchNorm2d(ngf * 4),
			nn.LeakyReLU(),

			nn.Upsample(scale_factor=2,  mode='nearest'),
			nn.Conv2d( in_channels = ngf * 4, out_channels = ngf * 2, kernel_size = 3, stride = 1, padding = 1, bias=False),
			nn.BatchNorm2d(ngf * 2),
			nn.LeakyReLU(),

			nn.Upsample(scale_factor=2,  mode='nearest'),
			nn.Conv2d( in_channels = ngf * 2, out_channels = 3, kernel_size = 3, stride = 1, padding = 1, bias=False),
			nn.Tanh()
        )

	def forward(self, input):
		bs = input.shape[0]
		input = input.view(bs,nz)
		output = self.linear(input)
		output = self.upsample(output.view(bs,1024, 4, 4))

		return output

######################################################################
######################################################################
######################################################################
######################################################################

class _netG_bilinear(nn.Module):
	def __init__(self):
		super(_netG_bilinear, self).__init__()

		self.linear = nn.Sequential(
			nn.Linear(100, ngf * 16 * 4 * 4),
			nn.BatchNorm1d(ngf * 16 * 4 * 4),
			nn.Dropout(p=p_dropout),
			nn.ReLU(),
			)

		self.upsample = nn.Sequential(
			nn.Upsample(scale_factor=2,  mode='bilinear'),
			nn.Conv2d( in_channels = ngf * 16, out_channels = ngf * 8, kernel_size = 3, stride = 1, padding = 1, bias=False),
			nn.BatchNorm2d(ngf * 8),
			nn.LeakyReLU(),

			nn.Upsample(scale_factor=2,  mode='bilinear'),
			nn.Conv2d( in_channels = ngf * 8, out_channels = ngf * 4, kernel_size = 3, stride = 1, padding = 1, bias=False),
			nn.BatchNorm2d(ngf * 4),
			nn.LeakyReLU(),

			nn.Upsample(scale_factor=2,  mode='bilinear'),
			nn.Conv2d( in_channels = ngf * 4, out_channels = ngf * 2, kernel_size = 3, stride = 1, padding = 1, bias=False),
			nn.BatchNorm2d(ngf * 2),
			nn.LeakyReLU(),

			nn.Upsample(scale_factor=2,  mode='bilinear'),
			nn.Conv2d( in_channels = ngf * 2, out_channels = 3, kernel_size = 3, stride = 1, padding = 1, bias=False),
			nn.Tanh()
        )

	def forward(self, input):
		bs = input.shape[0]
		input = input.view(bs,nz)
		output = self.linear(input)
		output = self.upsample(output.view(bs,1024, 4, 4))

		return output

######################################################################
######################################################################
######################################################################
######################################################################

class _netD(nn.Module):
	def __init__(self, method):
		super(_netD, self).__init__()

		self.method = method

		self.main = nn.Sequential(
			# input is (nc) x 64 x 64
			nn.Conv2d( in_channels = nc, out_channels = ndf, kernel_size = 4, stride = 2, padding = 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(in_channels = ndf, out_channels = ndf * 2, kernel_size = 4, stride = 2, padding = 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(in_channels = ndf * 2, out_channels = ndf * 4, kernel_size = 4, stride = 2, padding = 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 8 x 8
			nn.Conv2d(in_channels = ndf * 4, out_channels = ndf * 8, kernel_size = 4, stride = 2, padding = 1, bias=False),
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*8) x 4 x 4
			nn.Conv2d(in_channels = ndf * 8, out_channels = 1, kernel_size = 4, stride = 1, padding = 0, bias=False),
		)

		self.sigmoid = nn.Sigmoid()

	def forward(self, input):
		if self.method == 'GAN':
			output = self.sigmoid(self.main(input))
		if self.method == 'WGAN':
			output = self.main(input)

		return output.view(-1, 1).squeeze(1)