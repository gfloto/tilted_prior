import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import copy

import os
import sys
import util
import model
from scipy.special import hyp1f1 as M
from scipy.special import gamma as G

class Compute_weight(nn.Module):
    def __init__(self, loss_type, tilt, nz):
        super(Compute_weight, self).__init__()
        self.tilt = tilt
        self.nz = nz
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.loss_type = loss_type
        if tilt != None: 
            self.z_tilt = M(nz/2, 1/2, (tilt**2)/2) + tilt*np.sqrt(2) * G((nz+1)/2) / G(nz/2)\
                            * M((nz+1)/2, 3/2, (tilt**2)/2)

    def forward(self, x, x_out, mu, log_var, z):
        # p_x_z options
        if self.loss_type == 'l2':
            log_p_x_z = -torch.linalg.norm(x - x_out, dim=(1,2,3))
        elif self.loss_type == 'cross_entropy':    
            b = x.size(0)
            target = Variable(x.data.view(-1) * 255).long()
            x_out = x_out.contiguous().view(-1,256)
            x_out = self.ce_loss(x_out, target)
            log_p_x_z = -torch.sum(x_out.view(b, -1), 1)

        # p_z options
        if self.tilt == None:
            log_p_z = -torch.sum(z**2/2, 1) - self.nz/2*np.log(2*np.pi)
        else:
            log_p_z = -torch.sum(z**2/2, 1) + self.tilt*torch.linalg.norm(z, dim=1)\
                        - self.nz/2*np.log(2*np.pi) - np.log(self.z_tilt)

        # q_z_x
        z_eps = (z - mu) / torch.exp(0.5*logvar)
        log_q_z_x = -torch.sum(z_eps**2/2 + np.log(2*np.pi)/2\
                              + logvar/2 + np.log(nz)/2, 1)
        
        return log_p_x_z + log_p_z - log_q_z_x
        
def compute_nll(weights): 
    with torch.no_grad():
        NLL_loss = -(torch.log(torch.mean(torch.exp(weights - weights.max())))+weights.max()) 
        
    return NLL_loss


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='../ood/data', help='path to dataset')

    parser.add_argument('--samples', type=int, default=200, help='number of samples for OOD testing')
    parser.add_argument('--k', type=int, default=256, help='repeat for comute IWAE bounds')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--num_iter', type=int, default=100, help='number of iters to optimize')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')

    parser.add_argument('--test_name', default=None, help='path to model checkpoint')
    parser.add_argument('--aucroc', type=bool, default=True, help='boolean, run aucroc testing')
    
    opt = parser.parse_args()

    if opt.test_name == None:
        raise ValueError('enter a load folder')
 
    cudnn.benchmark = True
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    # information from test_name files
    load_path = os.path.join('results', opt.test_name)
    train_dataset, loss_type, tilt, nz = util.get_test_info(load_path)
    tilt = torch.tensor(float(tilt)) if tilt != None else None
    
    # display data, convert to useable datatypes
    print('latent dims: {}, tilt: {}'.format(nz, tilt))

    # load datasets
    batch_size = 1 # for regret step and IWAE batching
    print('loading data')
    
    loaders,loader_names,image_size = util.load_datasets(train_dataset, opt.dataroot,
                                                batch_size=1, num_workers=4)
    
    # load network and loss
    print('loading network') 
    netE = model.Encoder(image_size, nz, tilt)
    state_E = torch.load(os.path.join(load_path, 'encoder.pth'))
    netE.load_state_dict(state_E)
    netE.to(device)

    netD = model.Decoder(image_size, nz, loss_type)
    state_D = torch.load(os.path.join(load_path, 'decoder.pth'))
    netD.load_state_dict(state_D)
    netD.to(device)
        
    # class to compute IWAE nll, and standard loss function for backprop step
    loss_fn = model.Loss(loss_type, tilt, nz) 
    compute_weight = Compute_weight(loss_type, tilt, nz)

    # main test loop
    for n, loader in enumerate(loaders): 
        print('testing dataset: {}'.format(loader_names[n]))
        nll_track, regret_track = [], []
        
        for i, (xi, _) in enumerate(loader):
            # get negative log-likelihood from model
            x = xi.expand(opt.k,-1,-1,-1).contiguous()
            w_agg = []

            with torch.no_grad():
                for batch_number in range(1): 
                    x = x.to(device)
                    b = x.size(0) 
                
                    # network pass
                    z, mu, logvar = netE(x)
                    x_out = netD(z)
 
                    w = compute_weight(x, x_out, mu, logvar, z) 
                    w_agg.append(w)
                
                # negative log likelihood
                w_agg = torch.stack(w_agg).view(-1) 
                nll_before = compute_nll(w_agg) 

            # train copy of model under single data sample     
            xi = xi.to(device)
            b = xi.size(0)
            netE_copy = copy.deepcopy(netE)
            optimizer = optim.Adam(netE_copy.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=5e-5)
            target = Variable(xi.data.view(-1) * 255).long()
            for j in range(opt.num_iter):
                
                z, mu, logvar = netE_copy(xi)
                x_out = netD(z)

                recon, kld = loss_fn(xi, x_out, mu, logvar) 
                loss =  recon + kld

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # get new negative log-likelihood from model
            w_agg = []
            with torch.no_grad():
                x = xi.expand(opt.k,-1,-1,-1).contiguous()
                target = Variable(x.data.view(-1)*255).long()
                for batch_number in range(1):  
                    # network pass
                    z, mu, logvar = netE_copy(x) 
                    x_out = netD(z)
                    x_out.contiguous()

                    w = compute_weight(x, x_out, mu, logvar, z) 
                    w_agg.append(w)
                
                w_agg = torch.stack(w_agg).view(-1) 

                # negative log likelihood
                nll_after = compute_nll(w_agg)  
                regret = nll_before  - nll_after

                nll_track.append(nll_before.detach().cpu().numpy())
                regret_track.append(regret.detach().cpu().numpy())
                
            if i >= opt.samples or i == len(loader)-1: # test for n samples
                save_path = os.path.join(load_path, 'aucroc', 'regret')
                os.makedirs(save_path, exist_ok=True)
                np.save(os.path.join(save_path, loader_names[n] + ' score.npy'), regret_track) 
                break  

    if opt.aucroc:
        util.aucroc(opt.test_name, 'regret', loader_names, train_dataset)
