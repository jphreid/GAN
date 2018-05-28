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

from models import _netD
from models import _netG
from models import _netG_nearest
from models import _netG_bilinear

from utilities import weights_init
from utilities import grad_norm
from utilities import gan_label
from utilities import save_model
from utilities import calc_gradient_penalty
from utilities import data_celeb
from utilities import inception_score

def train(im_data, gen_model, method, name_file, niter, n_critic, lr, input_noise):
	nz = 100
	img_size = 64
	batch_size = 64
	beta1 = 0.5

	hyperparameters = {}
	hyperparameters['nz'] = nz
	hyperparameters['n_critic'] = n_critic
	hyperparameters['img_size'] = img_size
	hyperparameters['batch_size'] = batch_size
	hyperparameters['lr'] = lr
	hyperparameters['beta1'] = beta1

	dataloader = torch.utils.data.DataLoader(im_data, batch_size, shuffle=True)

	input = torch.FloatTensor(batch_size, 3, img_size, img_size)
	noise = torch.FloatTensor(batch_size, nz, 1, 1)
	fixed_noise = Variable(torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1))

	label = torch.FloatTensor(batch_size)
	
	# real_label = 1
	# fake_label = 0

	if gen_model == 'nearest':
		netG = _netG_nearest()
	elif gen_model == 'bilinear':
		netG = _netG_bilinear()
	elif gen_model == 'transposed_conv2d':
		netG = _netG()
	
	netG.apply(weights_init)

	if method =='GAN':
		netD = _netD(method)
	elif method =='WGAN':	
		netD = _netD(method)	

	netD.apply(weights_init)
	criterion = nn.BCEWithLogitsLoss()
	
	if method == 'GAN':
		optimizerD = optim.Adam(netD.parameters(), lr, betas=(beta1, 0.9))
		optimizerG = optim.Adam(netG.parameters(), lr, betas=(beta1, 0.9))

	elif method == 'WGAN':
		optimizerD = optim.RMSprop(netD.parameters(), lr=lr)
		optimizerG = optim.RMSprop(netG.parameters(), lr=lr)

	if torch.cuda.is_available():
		netD.cuda()
		netG.cuda()
		criterion.cuda()
		input, label = input.cuda(), label.cuda()
		noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
	
	epochl = []

	wdl = []
	errDml = []
	errGml = []

	errDsl = []
	errGsl = []
	
	errorGl = []
	errorDl = []
	
	dxl = []
	dgz1l = []
	dgz2l = []

	grad_netDl = []
	grad_netGl = []

	for epoch in range(niter):
		errDm = []
		errGm = []
		wdm = []
		for i, data in enumerate(dataloader):

			for j in range(n_critic):
				#############################################################
				# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
				#############################################################

				# train with real
				netD.zero_grad()

				# #Might want to add some gaussian noise to the data. This is where it's happening. 
				if input_noise=='yes':
					sigma = 0.1
					gaussian_noise = data.new(data.size()).normal_(0, sigma)
					normalize = torch.max(torch.abs(data + gaussian_noise))
					real_cpu = (data + gaussian_noise)/normalize

				else:
					real_cpu = data


				batch_size = real_cpu.size(0)
				if torch.cuda.is_available():
					real_cpu = real_cpu.cuda()

				# train with real
				input.resize_as_(real_cpu).copy_(real_cpu)
				real_label = gan_label(1,'D')
				label.resize_(batch_size).fill_(real_label)

				inputv = Variable(input)
				labelv = Variable(label)

				output = netD(inputv)
				if method == 'GAN':
					errD_real = torch.log(output)#criterion(output, labelv) # labelv = real
				elif method == 'WGAN':
					errD_real = torch.mean(output)

				D_x = output.data.mean()

				# train with fake
				noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
				noisev = Variable(noise)
				fake = netG(noisev)
				fake_label = gan_label(0,'D')
				labelv = Variable(label.fill_(fake_label)) # 0
				output = netD(fake.detach())
				if method == 'GAN':
					errD_fake = torch.log(1- output) #criterion(output, labelv) # labelv = fake
				elif method == 'WGAN':
					errD_fake = torch.mean(output)
				
				D_G_z1 = output.data.mean()
	
				grad_penalty = calc_gradient_penalty(netD, inputv, fake)

				if method == 'GAN': 
					errD = -torch.mean(errD_real + errD_fake) + grad_penalty

				if method == 'WGAN':
					errD = -(torch.mean(errD_real) - torch.mean(errD_fake)) + grad_penalty

				errD.backward()
				optimizerD.step()

				if method == 'GAN':
					pass
				if method == 'WGAN':
					for p in netD.parameters():
						p.data.clamp_(-0.05, 0.05)

				wd = torch.mean(errD_real - errD_fake)
			wdm.append(wd.data[0])
			errDm.append(errD.data[0])
			
			#############################################
			# (2) Update G network: maximize log(D(G(z)))
			#############################################

			netG.zero_grad()
			real_label = gan_label(1, 'G')
			labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
			output = netD(fake) # between 0, 1 ... it'S a good cop! because, we trained above! 
			
			if method == 'GAN':
				errG = -torch.mean(torch.log(output)) #criterion(output, labelv) # labelv = real
			elif method == 'WGAN':
				errG = -torch.mean(output)

			errG.backward()
			errGm.append(errG.data[0])

			D_G_z2 = output.data.mean()
			optimizerG.step()

			grad_netD = grad_norm(netD)
			grad_netG = grad_norm(netG)

			# torch.nn.utils.clip_grad_norm(netG.parameters(), 0.2, norm_type=2)

			print('[{}/{}][{}/{}] Loss_D: {:8f} Loss_G: {:4f} D(x): {:4f} D(G(z)): {:4f} / {:4f} Grad_D: {:2f} Grad_D: {:2f}'.format(
							epoch+1, niter, i+1, len(dataloader),errD.data[0], 
							errG.data[0], D_x, D_G_z1, D_G_z2, grad_netD, grad_netG))
			

			errorDl.append(errD.data[0])
			errorGl.append(errG.data[0])
			dxl.append(D_x)
			dgz1l.append(D_G_z1)
			dgz2l.append(D_G_z2)
			epochl.append(epoch)
			grad_netDl.append(grad_netD)
			grad_netGl.append(grad_netG)

		#print at the end of each epoch. 
		fake = netG(fixed_noise)
		vutils.save_image(fake.data, 'savedata/figures/{}_{}_fake_samples_epoch_{}_i_{}.png'.format(gen_model, method, epoch, i), normalize=True)
		# vutils.save_image(real_cpu,'figures/outputs/{}_real_samples_epoch_{}_i_{}.png'.format(gen_model, epoch, i), normalize=True)

		wdl.append(np.mean(np.array(wdm)))
		errDml.append(np.mean(np.array(errDm)))
		errGml.append(np.mean(np.array(errGm)))
		errDsl.append(np.std(np.array(errDm)))
		errGsl.append(np.std(np.array(errGm)))
		
		torch.save(netG.state_dict(), 'savedata/models/{}_{}_netG_epoch_{}.pth'.format(name_file, method, epoch))
		torch.save(netD.state_dict(), 'savedata/models/{}_{}_netD_epoch_{}.pth'.format(name_file, method, epoch))
	
		if epoch%1 == 0 : 
			dd = {}
			if input_noise=='yes':
				dd['sigma'] = sigma
			else:
				pass
			dd['wd'] = wdl
			dd['gen_model'] = gen_model
			dd['epoch'] = epochl
			dd['errDm'] = errDml
			dd['errGm'] = errGml
			dd['errDs'] = errDsl
			dd['errGs'] = errGsl
			dd['error_d'] = errorDl
			dd['error_g'] = errorGl
			dd['dx'] = dxl
			dd['dgz1'] = dgz1l
			dd['dgz2'] = dgz2l
			dd['grad_netD'] = grad_netDl
			dd['grad_netG'] = grad_netGl
		
			filename = 'savedata/data/{}_epoch_{}.pkl'.format(name_file, epoch)
			with open(filename, 'wb') as  f: 
				pickle.dump([hyperparameters, dd], f)

	dd = {}
	if input_noise=='yes':
		dd['sigma'] = sigma
	else:
		pass
	dd['wd'] = wdl
	dd['gen_model'] = gen_model
	dd['epoch'] = epochl
	dd['errDm'] = errDml
	dd['errGm'] = errGml
	dd['errDs'] = errDsl
	dd['errGs'] = errGsl
	dd['error_d'] = errorDl
	dd['error_g'] = errorGl
	dd['dx'] = dxl
	dd['dgz1'] = dgz1l
	dd['dgz2'] = dgz2l
	dd['grad_netD'] = grad_netDl
	dd['grad_netG'] = grad_netGl		
	filename = 'savedata/data/{}_epoch_{}.pkl'.format(name_file, epoch)
	with open(filename, 'wb') as  f: 
		pickle.dump([hyperparameters, dd], f)
	print()
	print(dd['wd'])