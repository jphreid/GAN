import numpy as np
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
from torchvision.models.inception import inception_v3


def data_celeb(n):
	''' n = 1 will load all the data'''
	print('Loading 1/{} of the data -- this may take a while'.format(n))
	t0 = time.time()
	path = 'data/resized_celebA/celebA/'
	im_data = []
	for root, _, filenames in scandir.walk(path):
		for i, f in enumerate(filenames):
			if float(f[:-4]) % n == 0:
				im = Image.open(root+f)
				im = im.convert('RGB')
				im = list(im.getdata())
				im_data.append(im)
	return im_data


def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

def grad_norm(model):
        total_norm=0
        for p in list(model.parameters()):
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm

        return total_norm

def gan_label(_label, D_G):
		if D_G == 'D':
			if _label == 1:
				label = np.random.binomial(1, 0.98, 1)[0].astype(float)
			elif _label == 0: 
				label = np.random.binomial(1, 0.02, 1)[0].astype(float)

		elif D_G == 'G':
			label = _label

		return label

def save_model(m, e_, m_name):
		state_dict = m.state_dict()
		for key in state_dict.keys():
			state_dict[key] = state_dict[key].cpu()
        
		torch.save({
			'epoch': e_,
			'state_dict': state_dict},
			m_name)


def calc_gradient_penalty(netD, real_data, fake_data):#netD, real_data, fake_data
	# print "real_data: ", real_data.size(), fake_data.size()
	n_batch = real_data.shape[0]
	lambda_ = 10
	alpha = torch.rand(n_batch, 1)
	n = real_data.nelement()
	
	alpha = torch.Tensor((alpha.expand(n_batch, int(n/n_batch)).contiguous().view(n_batch, 3, 64, 64)))

	if torch.cuda.is_available():
		alpha = alpha.cuda()

	interpolates = alpha * real_data.data + ((1 - alpha) * fake_data.data)
	
	if torch.cuda.is_available():
		interpolates = interpolates.cuda()

	interpolates = Variable(interpolates, requires_grad=True)
	disc_interpolates = netD(interpolates)
	grad_outputs = torch.ones(disc_interpolates.size())

	if torch.cuda.is_available():
		grad_outputs = torch.ones(disc_interpolates.size()).cuda()

	gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
									grad_outputs=grad_outputs,
									create_graph=True, retain_graph=True, only_inputs=True)[0]

	gradients = gradients.view(gradients.size(0), -1)
	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_

	return gradient_penalty

from torchvision.models.inception import inception_v3
from torch.nn import functional as F
from scipy.stats import entropy


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)
    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]
        
        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

