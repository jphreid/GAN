import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import pickle
import time
import scandir

import os
from os import listdir
from os.path import isfile, join
from PIL import Image

from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from scipy.stats import entropy

######################################################################
######################################################################
######################################################################
######################################################################
######################################################################

######################################################################
######################################################################
######################################################################
######################################################################

#Size of latnet vector
nz = 100
# Filter size of generator
ngf = 64
# Filter size of discriminator
ndf = 64
# Output image channels
nc = 3
# Batch size
p_dropout = 0.5

from models import _netD
from models import _netG
from models import _netG_nearest
from models import _netG_bilinear

from train_model import train

from utilities import weights_init
from utilities import grad_norm
from utilities import gan_label
from utilities import save_model
from utilities import calc_gradient_penalty
from utilities import data_celeb
from utilities import inception_score

######################################################################
######################################################################
######################################################################
######################################################################


if __name__ == "__main__":
	'''loading 1/dd of the data'''
	dd = 20
	im_data = data_celeb(dd)

	# mean calculated on 1/5 of the data 
	mean = torch.Tensor([0.5064, 0.4260, 0.3831]).view(1,1,3)
	std = torch.Tensor([0.3040, 0.2835, 0.2831]).view(1,1,3)

	# Normalize
	im_data = torch.Tensor(im_data)/255
	# Standardize
	im_data_n = (im_data - mean)/std
	im_data_n = im_data_n.view(len(im_data), 64, 64, 3).permute(0,3,1,2)

	'''
	Train
	niter : number of epochs
	name_file : file name in which the data is saved
	gen_model : different netG model - Q3
	'''

	'''Q4 (dd = 10)''' 
	nmax = 20
	n_critic = 5

	# lr = 5e-5
	# method = 'WGAN'
	# filename = 'experience_{}_transposed_conv2d_nc={}_same_optimizer_lr_{}_n_{}_epoch_{}'.format(method, n_critic, lr, dd, nmax)
	# train(im_data_n, gen_model='transposed_conv2d', name_file=filename, method=method, niter=nmax, n_critic=n_critic, lr=lr, input_noise='yes')

	lr = 1e-5
	method = 'GAN'
	filename = 'experience_{}_transposed_conv2d_nc={}_same_optimizer_lr_{}_n_{}'.format(method, n_critic, lr, dd)
	train(im_data_n, gen_model='transposed_conv2d', name_file=filename, method=method, niter=nmax, n_critic=n_critic, lr=lr, input_noise='yes')

	'''Q3 (dd = 20)'''
	# nmax = 60
	# n_critic = 5

	# lr = 1e-5
	# method = 'GAN'
	# filename = 'experience_{}_bilinear_nc={}_same_optimizer_lr_{}_n_{}'.format(method, n_critic, lr, dd)
	# train(im_data_n, gen_model='bilinear', name_file=filename, method=method, niter=nmax, n_critic=n_critic, lr=lr, input_noise='yes')

	# lr = 1e-5
	# method = 'GAN'
	# filename = 'experience_{}_nearest_nc={}_same_optimizer_lr_{}_n_{}'.format(method, n_critic, lr, dd)
	# train(im_data_n, gen_model='nearest', name_file=filename, method=method, niter=nmax, n_critic=n_critic, lr=lr, input_noise='yes')
 	
	# lr = 1e-5
	# method = 'GAN'
	# filename = 'experience_{}_transposed_conv2d_nc={}_same_optimizer_lr_{}_n_{}'.format(method, n_critic, lr, dd)
	# train(im_data_n, gen_model='transposed_conv2d', name_file=filename, method=method, niter=nmax, n_critic=n_critic, lr=lr, input_noise='yes')


	##################################################################################################
	# The lines are useful if you want to understand the forward pass of _netD() and _netG(). 
	##################################################################################################

	#############################################
	############################################

	# sigmoid = nn.Sigmoid()
	# # criterion = nn.BCELoss()
	# criterion = nn.BCEWithLogitsLoss()
	# label = torch.Tensor(1)

	# '''D'''
	# netD = _netD('GAN')
	# netD.apply(weights_init)
	# batch_example = Variable(im_data_n[:1])
	# pred = netD(batch_example)
	# labelv = Variable(label.fill_(1))
	# errD = criterion(pred, labelv)
	
	# '''G'''
	# netG = _netG_bilinear()
	# netG.apply(weights_init)
	# noise = torch.FloatTensor(64, nz, 1)
	# noise.resize_(64, 100, 1, 1).normal_(0, 1)
	# noisev = Variable(noise)
	# pred = netD(netG(noisev))[1]
	# labelv = Variable(label.fill_(1))
	# errG = criterion(pred, labelv)

	# grad_penalty = calc_gradient_penalty(netD, im_data_n[:1], netG(noisev).data)


	## To study differences between BCEloss and BCEwith... 
	
	# sigmoid = nn.Sigmoid()
	# criterion1 = nn.BCELoss()
	# criterion2 = nn.BCEWithLogitsLoss()

	# real_label = 1
	# p = 0.8
	# label = torch.Tensor(1)
	# pred = torch.Tensor(1)
	# labelv = Variable(label.fill_(real_label))
	# pv = Variable(pred.fill_(p))

	# # def criterion_smoothing_labels():
	# p = pv
	# class1 = 0.9
	# yhat = 0.9
	# ll = (yhat/class1) * torch.log(p) + (1 - (class1/yhat)) * torch.log(1 - p)
	
	# print(-ll)
	# print(criterion1(pv, labelv))
	
	# print(criterion2(sigmoid(pv), labelv))
