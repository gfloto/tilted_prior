import os
import sys
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

import util
import model

######## changes
# net conv structure (for l2)
# net batch norm removed
# l2 loss in train loop
# include latent dimension dependence in IWAE
# sample images nz encoder 

# for likelihood ratio
def perturb(x, mu, device):
    b, c, h, w = x.size()
    mask = torch.rand(b, c, h, w) < mu
    mask = mask.float().to(device)
    noise = torch.FloatTensor(x.size()).random_(0, 256).to(device)
    x = 255*x
    perturbed_x = ((1 - mask)*x + mask*noise)/255.
    
    return perturbed_x

if __name__=="__main__":
    cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='../ood/data', help='path to dataset')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--nz', type=int, default=10, help='size of the latent z vector')
    parser.add_argument('--epochs', type=int, default=250, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--beta', type=float, default=1., help='beta for beta-vae')

    parser.add_argument('--perturbed', type=bool, default=False, help='Whether to train on perturbed data, used for comparing with likelihood ratio by Ren et al.')
    parser.add_argument('--ratio', type=float, default=0.2, help='ratio for perturbation of data, see Ren et al.')

    parser.add_argument('--test_name', default=None)
    parser.add_argument('--dataset', default='fmnist', help='train dataset, either fmnist or cifar10')
    parser.add_argument('--loss', default='l2', help='loss, either: cross_entropy or l2')
    parser.add_argument('--tilt', default=None, help='tilt, if None: regular vae w learnable variance')

    parser.add_argument('--images', type=bool, default=True, help='boolean, sample images')
    parser.add_argument('--burn_in', type=bool, default=False, help='train vae with reverse beta annealing and decoder burn in')

    opt = parser.parse_args()
    save_path = os.path.join('results', opt.test_name)

    if opt.test_name == None:
        raise ValueError('enter a test name')
     
    # set random seed 
    opt.manualSeed = random.randint(1, 10000)
    print("random seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed) 

    # set device
    if torch.cuda.is_available():
        device = 'cuda'
        print('using gpu')
    else:
        device = 'cpu'
        print('using cpu')

    # load train distributions
    print('loading data')
    if opt.dataset == 'fmnist':
        print('using fmnist')
        image_size = [32, 32, 1]
        dataset_fmnist_train = dset.FashionMNIST(root=opt.dataroot, train=True, download=True, transform=transforms.Compose([ 
                                    transforms.ToTensor(),
                                    transforms.Resize((image_size[0])),
                                ]))
        dataloader = torch.utils.data.DataLoader(dataset_fmnist_train, batch_size=opt.batch_size,
                                                shuffle=True, num_workers=int(opt.workers)) 

    elif opt.dataset == 'cifar10':
        print('using cifar10')
        image_size = [32, 32, 3]
        dataset = dset.CIFAR10(root=opt.dataroot, download=True,train = True,
                                transform=transforms.Compose([
                                    transforms.Resize((image_size[0])),
                                    transforms.ToTensor(),
                                ]))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                                shuffle=True, num_workers=int(opt.workers))

    elif opt.dataset == 'celeba':
        print('using celeba')
        image_size = [178, 178, 3]
        celeba_set = util.CelebaDataset(txt_path=opt.dataroot+'/celeba/identity_CelebA.txt',
                                    img_dir=opt.dataroot+'/celeba/img_align_celeba/',
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.CenterCrop((176,176))]),
                                        train=True
                                    ) 

        dataloader = data.DataLoader(celeba_set, batch_size=opt.batch_size, 
                                        shuffle=True, num_workers=int(opt.workers))
        
    else:
        raise ValueError('{} is not a valid dataset to train on, choose fmnist or cifar10'.format(opt.dataset))

    # weight initialization
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        
    # display data, convert to useable datatypes
    print('latent dims: {}, tilt: {}, burn in: {}'.format(opt.nz, opt.tilt, opt.burn_in))
    nz = int(opt.nz)
    tilt = torch.tensor(float(opt.tilt)) if opt.tilt != None else None

    loss_fn = model.Loss(opt.loss, tilt, nz)
    max_grad_norm = 100

    # load network and loss function
    print('loading network')  
    netE = model.Encoder(image_size, nz, loss_fn.gamma)
    netE.apply(weights_init)
    netE.to(device)

    netD = model.Decoder(image_size, nz, opt.loss)
    netD.apply(weights_init)
    netD.to(device) 

    # setup optimizers 
    weight_decay = 3e-5
    optimizer1 = optim.Adam(netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=weight_decay)
    optimizer2 = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=weight_decay) 
    
    # create file with training information
    util.info_file(opt)

    # decoder burn in
    if opt.burn_in == True:
        print('decoder burn in')
        opt.beta = 0.00001

        for x, _ in dataloader:
            x = x.to(device)

            z = torch.randn((x.shape[0], nz)).to(device)
            x_out = netD(z)

            if opt.loss == 'l2':
                recon = torch.linalg.norm(x - x_out, dim=(1,2,3))
                loss = torch.mean(recon)
            else: # else cross_entropy
                b = x.size(0)
                target = Variable(x.data.view(-1) * 255).long()
                out = x_out.contiguous().view(-1,256)
                recon = F.cross_entropy(out, target, reduction='none')
                loss = torch.sum(recon) / b

            # optimize
            optimizer2.zero_grad()

            loss.backward()
            for group in optimizer2.param_groups: # clip gradients
                utils.clip_grad_norm_(group['params'], max_grad_norm, norm_type=2)
            
            optimizer2.step()


    # main training look
    recon_track, kld_track = [], [] # arrays for storing loss information
    for epoch in range(opt.epochs):
        recon_temp, kld_temp = [], [] # arrays for storing loss information
        for i, (x, _) in enumerate(dataloader):            
            x = x.to(device)
             
            # for likelihood ratio
            if opt.perturbed:
                x = perturb(x, opt.ratio, device)

            # main network pass 
            z, mu, logvar = netE(x)
            x_out = netD(z)
            
            recon, kld = loss_fn(x, x_out, mu, logvar)
            loss = recon + opt.beta * kld
             
            # optimize
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()
            for group in optimizer1.param_groups: # clip gradients
                utils.clip_grad_norm_(group['params'], max_grad_norm, norm_type=2)
            for group in optimizer2.param_groups: # clip gradients
                utils.clip_grad_norm_(group['params'], max_grad_norm, norm_type=2)

            optimizer1.step()
            optimizer2.step()
                       
            recon_temp.append(recon.mean().detach().item())
            kld_temp.append(kld.mean().detach().item())

            # store and print info
            if not i % 100:
                print('epoch:{} recon:{:.3f} kld:{:.3f}'.format(
                        epoch, np.mean(recon_temp), np.mean(kld_temp)))

        # beta-annealing
        if opt.burn_in and opt.beta < 1 and (epoch+1)%10 == 0:
            opt.beta *= 2
        if opt.beta > 1:
            opt.beta = 1

        recon_track.append(np.mean(recon_temp))
        kld_track.append(np.mean(kld_temp))

        
        if epoch%2 == 0 or epoch == opt.epochs-1:
            # save models        
            torch.save(netE.state_dict(), os.path.join(save_path, 'encoder.pth'))
            torch.save(netD.state_dict(), os.path.join(save_path, 'decoder.pth'))

            # save and plot loss data
            util.loss_plot(save_path, recon_track, kld_track)  

            # save sample images from training set and OOD
            if opt.images:
                if opt.dataset != 'celeba':
                    loaders, names, _ = util.load_datasets(opt.dataset, opt.dataroot,
                                                            batch_size=10, num_workers=4)
                else:
                    loaders = [dataloader]
                    names = ['celeba']

                util.sample_images(netE, netD, loaders, names, save_path, device)
