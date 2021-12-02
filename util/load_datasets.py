import torch
import torchvision
import torch.utils.data as data
from torchvision import transforms

import os
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# custom dataloader for celeba
class CelebaDataset(Dataset):
    def __init__(self, txt_path, img_dir, transform, train=False):
        df = pd.read_csv(txt_path, sep=" ", index_col=0)
        self.img_dir = img_dir
        self.txt_path = txt_path
        self.img_names = df.index.values
        self.y = df['attr'].values
        self.transform = transform
        self.len = 50000 if not train else 202599
        # when testing ood use first 50 000 imgs (same size as cifar-10)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))
        
        img = self.transform(img)
        label = self.y[index]
        return img, label

    def __len__(self):
        return self.len

# custom dataloader for noise and constant data
class FakeData(Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.transform = transform

    def __getitem__(self, index):
        img = np.load(os.path.join(self.path, "img {}.npy".format(index)),
                      allow_pickle=True)        
        img = self.transform(img)
        label = 0
        return img, label

    def __len__(self):
        return 60000 # same size as mnist type dataset

# custom transforms
class Gray2Color(object):
    def __call__(self, img):
        img = torch.vstack((img, img, img)) 
        return img # stack three channels  

class Color2Gray(object):
    def __call__(self, img):
        return torch.unsqueeze(img[0], axis=0) # take first channel of image

def load_datasets(dataset, root, batch_size, num_workers=2):
    # train datasets
    train_datasets = ['fmnist', 'cifar10']
    if dataset not in train_datasets:
        print('invalid train dataset: ', dataset)
        return

    if dataset == 'fmnist':
        # transformations
        c2g = Color2Gray()
        transform = transforms.Compose([transforms.ToTensor()])
        gray_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32,32))])
        color_transform = transforms.Compose([transforms.ToTensor(), c2g])

        # datasets
        fmnist_set = torchvision.datasets.FashionMNIST(root=root, train=True, download=False, transform=gray_transform)
        fmnist_loader = data.DataLoader(fmnist_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        mnist_set = torchvision.datasets.MNIST(root=root, train=True, download=False, transform=gray_transform)
        mnist_loader = data.DataLoader(mnist_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        cifar10_set = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=color_transform)
        cifar10_loader = data.DataLoader(cifar10_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        svhn_set = torchvision.datasets.SVHN(root=root, download=False, transform=color_transform)
        svhn_loader = data.DataLoader(svhn_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        kmnist_set = torchvision.datasets.KMNIST(root=root, train=True, download=False, transform=gray_transform)
        kmnist_loader = data.DataLoader(kmnist_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        uniform_set = FakeData(path=root+'/fake/uniform_gray32', transform=transform)
        uniform_loader = data.DataLoader(uniform_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        const_set = FakeData(path=root+'/fake/const_gray32', transform=transform)
        const_loader = data.DataLoader(const_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        loader_names = ['fmnist', 'mnist', 'cifar10', 'svhn', 'kmnist', 'noise', 'constant']
        loaders = [fmnist_loader, mnist_loader, cifar10_loader, svhn_loader, kmnist_loader, uniform_loader, const_loader]
        image_size = [32, 32, 1]
        return loaders, loader_names, image_size

    elif dataset == 'cifar10':
        # transformations
        g2c = Gray2Color()
        transform = transforms.Compose([transforms.ToTensor()])
        gray_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32,32)), g2c])
        lsun_transform = transforms.Compose([transforms.ToTensor(),
                            transforms.CenterCrop((256,256)),
                            transforms.Resize((32,32))])
        celeba_transform = transforms.Compose([transforms.ToTensor(),
                            transforms.CenterCrop((178,178)),
                            transforms.Resize((32,32))]) 

        # datasets
        cifar10_set = torchvision.datasets.CIFAR10(root=root, train=True, download=False, transform=transform)
        cifar10_loader = data.DataLoader(cifar10_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        mnist_set = torchvision.datasets.MNIST(root=root, train=True, download=False, transform=gray_transform)
        mnist_loader = data.DataLoader(mnist_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        fmnist_set = torchvision.datasets.FashionMNIST(root=root, train=True, download=False, transform=gray_transform)
        fmnist_loader = data.DataLoader(fmnist_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        svhn_set = torchvision.datasets.SVHN(root=root, download=False, transform=transform)
        svhn_loader = data.DataLoader(svhn_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        lsun_set = torchvision.datasets.LSUN(root=root+'/lsun', classes='val', transform=lsun_transform)
        lsun_loader = data.DataLoader(lsun_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
        celeba_set = CelebaDataset(txt_path=root+'/celeba/identity_CelebA.txt', img_dir=root+'/celeba/img_align_celeba/', transform=celeba_transform)
        celeba_loader = data.DataLoader(celeba_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        uniform_set = FakeData(path=root+'/fake/uniform_color32', transform=transform)
        uniform_loader = data.DataLoader(uniform_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        const_set = FakeData(path=root+'/fake/const_color32', transform=transform)
        const_loader = data.DataLoader(const_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # not mnist
        loader_names = ['cifar10', 'mnist', 'fmnist', 'svhn', 'lsun', 'celeba', 'noise', 'constant']
        loaders = [cifar10_loader, mnist_loader, fmnist_loader, svhn_loader, lsun_loader, celeba_loader, uniform_loader, const_loader]
        im_shape = (32, 32, 3)
        return loaders, loader_names, im_shape
 