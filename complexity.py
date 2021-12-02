import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

import os
import cv2
import model
import util
from scipy.special import hyp1f1 as M
from scipy.special import eval_genlaguerre as L 
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
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')

    parser.add_argument('--test_name', default=None, help='path to model checkpoint')
    parser.add_argument('--ic_type', default='png', help='type of complexity measure, choose between png and jp2')
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
    save_name = 'complexity ' + opt.ic_type if opt.ic_type != 'none' else 'nll'
    
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

    # class to compute IWAE nll
    compute_weight = Compute_weight(loss_type, tilt, nz)

    # main test loop 
    for n, loader in enumerate(loaders):
        print('testing dataset: {}'.format(loader_names[n]))
        score_track = [] # to track finaly complexity score

        for i, (x, _) in enumerate(loader):
            # get negative log-likelihood from model
            x = x.expand(opt.k,-1,-1,-1).contiguous()  
            w_agg  = []

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
                nll = compute_nll(w_agg)

                # convert for image compressor step 
                img = x[0].permute(1,2,0)
                img = img.detach().cpu().numpy()
                img *= 255
                img = img.astype(np.uint8)
                
                if opt.ic_type == 'jp2':
                    img_encoded = cv2.imencode('.jp2',img)
                elif opt.ic_type == 'png':
                    img_encoded = cv2.imencode('.png',img,[int(cv2.IMWRITE_PNG_COMPRESSION),9])
                elif opt.ic_type == 'none':
                    # for pure nll with IWAE
                    pass
                else:
                    raise NotImplementedError("choose ic type between jp2, png and none")
                
                if opt.ic_type != 'none':
                    complexity = len(img_encoded[1]) * 8 # 8-bit encoding 
                    score_track.append(nll.detach().cpu().numpy() - complexity)
                else:
                    score_track.append(nll.detach().cpu().numpy())


            if i >= opt.samples or i == len(loader)-1:
                save_path = os.path.join(load_path, 'aucroc', save_name)
                os.makedirs(save_path, exist_ok=True)
                np.save(os.path.join(save_path, loader_names[n] + ' score.npy'), score_track) 
                break        
  
    if opt.aucroc:
        util.aucroc(opt.test_name, save_name, loader_names, train_dataset)
