import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

# plot auroc and distributions
def aucroc(test_name, test_type, datasets, train_dataset):
    load_path = os.path.join('results', test_name, 'aucroc', test_type)

    # always plot training dist. last in the same color
    name = datasets.copy()
    name.insert(len(name)-1, name.pop(name.index(train_dataset)))
    
    plt.style.use('seaborn')    
    colors = sns.color_palette('Set3', len(datasets) + 4)
    figure = plt.figure(figsize=(6,6))

    score = []
    for n in name:
        score.append(np.load(os.path.join(load_path, '{} score.npy'
                        .format(n))))

    # calculate and plot aucroc
    aucroc_track = []
    label_1 = np.ones(score[-1].shape[0])
    for i in range(len(score) - 1):
        combined = np.concatenate((score[-1], score[i]))
        label_2 = np.zeros(score[i].shape[0])
        label = np.concatenate((label_1, label_2))

        fpr, tpr, thresholds = metrics.roc_curve(label, combined, pos_label=0)
        auc = metrics.auc(fpr, tpr)
        aucroc_track.append(auc)

        # plot
        plt.plot(fpr, tpr, color=colors[i])
        print('aucroc on {}: '.format(name[i]), auc)
        name[i] += ': {:.3}'.format(auc)
    
    np.save(os.path.join(load_path, 'aucroc_info.npy'), aucroc_track)

    plt.plot([0,1], [0,1], color='gray', linestyle='--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('{} AUCROC'.format(train_dataset.upper()))
    plt.legend(name[:-1])

    plt.tight_layout()
    plt.savefig(os.path.join(load_path, 'aucroc_plot.pdf'))
    plt.close()

	# save text file with aucroc info
    file = open(os.path.join(load_path, 'score.txt'), 'w')
    for n in name:
        file.write('{}\n'.format(n))
    file.close()

    # plot distributions
    bins = 20
    figure = plt.figure(figsize=(6,6))
    for i, sc in enumerate(score):
        sns.histplot(sc, bins=bins, color=colors[i])
    
    plt.ylabel('count')
    plt.xlabel('score')
    plt.title('{} Histogram'.format(train_dataset.upper()))
    plt.legend(name)
    plt.savefig(os.path.join(load_path, 'histogram.pdf'))

# plot training loss
def loss_plot(save_path, recon_track, kld_track):
    recon_track = np.array(recon_track)
    kld_track = np.array(kld_track)
    plt.style.use('seaborn')

    fig = plt.figure(figsize=(18,6))
    fig1 = fig.add_subplot(131)
    fig2 = fig.add_subplot(132)
    fig3 = fig.add_subplot(133)

    fig1.plot(recon_track, 'k')
    fig2.plot(kld_track, 'r')
    fig2.set_yscale('log')
    fig3.plot(recon_track + kld_track, 'b')

    fig1.set_title('Reconstruction')
    fig2.set_title('KL-Divergence')
    fig3.set_title('Total Loss')

    fig2.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,'loss.pdf'))
    plt.close()

    np.save(os.path.join(save_path, 'recon.npy'), recon_track)
    np.save(os.path.join(save_path, 'kld.npy'), kld_track)

# save sample images
def sample_images(netE, netD, loaders, name, save_path, device, num=5):
    # save folder 
    save_path = os.path.join(save_path, 'images')
    os.makedirs(save_path, exist_ok=True)

    # test in distribution and OOD 
    for i, loader in enumerate(loaders):
        for x, _ in loader:
            x = x.to(device)

            # reconstruction 
            z, _, _ = netE(x)
            x_out = netD(z)
 
            # sample
            if netE.gamma == None:
                z_sample = torch.randn_like(z)
            else:
                nz = netE.nz
                z_sample = torch.randn_like(z, dtype=torch.float32, device=device)
                z /= torch.outer(torch.linalg.norm(z_sample, dim=(1)),
                                 torch.ones(z_sample.shape[1]).to(device)) 
                z *= netE.gamma
            
            x_sample = netD(z_sample)

            # correct for cross entropy output
            if netD.loss_type == 'cross_entropy':
                x_out = torch.argmax(x_out, dim=4)/255
                x_sample = torch.argmax(x_sample, dim=4)/255

            # clip for displaying
            x_out = torch.clip(x_out, 0, 1)
            x_sample = torch.clip(x_sample, 0, 1)

            # plot and save
            for j in range(num):
                fig = plt.figure(figsize=(15,5))
                fig1 = fig.add_subplot(131)
                fig2 = fig.add_subplot(132)
                fig3 = fig.add_subplot(133)

                fig1.axis('off')
                fig2.axis('off')
                fig3.axis('off')
    
                fig1.set_title('Input')
                fig2.set_title('Output') 
                fig3.set_title('Sample')

                # plot
                # gray images
                if x.shape[1] == 1:
                    s = x.shape
                    fig1.imshow(x[j].detach().cpu().numpy().reshape(s[2],s[3]),
                                    cmap='gray', vmin=0, vmax=1)
                    fig2.imshow(x_out[j].detach().cpu().numpy().reshape(s[2],s[3]),
                                    cmap='gray', vmin=0, vmax=1)
                    fig3.imshow(x_sample[j].detach().cpu().numpy().reshape(s[2],s[3]),
                                    cmap='gray', vmin=0, vmax=1)
                # color images
                else:
                    fig1.imshow(x[j].permute(1,2,0).detach().cpu().numpy(),
                                    vmin=0, vmax=1)
                    fig2.imshow(x_out[j].permute(1,2,0).detach().cpu().numpy(),
                                    vmin=0, vmax=1)
                    fig3.imshow(x_sample[j].permute(1,2,0).detach().cpu().numpy(),
                                    vmin=0, vmax=1)
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, '{} {}.png'.format(name[i], j)))
                plt.close()
        
            # one batch per loader
            break
