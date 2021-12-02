import os

# save opt to file
def info_file(opt):
    path = os.path.join('results', opt.test_name)
    os.makedirs(path, exist_ok=True)
        
    file = open(os.path.join(path, 'information.txt'), 'w')
    file.write('train dataset: {}\n'.format(opt.dataset))
    file.write('batch size: {}\n'.format(opt.batch_size))
    file.write('learning rate: {}\n'.format(opt.lr))
    file.write('loss type: {}\n'.format(opt.loss))
    file.write('tilt: {}\n'.format(opt.tilt))
    file.write('latent dim: {}\n'.format(opt.nz))
    file.write('burn in and beta annealing: {}\n'.format(opt.burn_in))
    file.close()

# return info for testing
def get_test_info(path):
    file = open(os.path.join(path, 'information.txt'))
    lines = file.readlines()

    for line in lines:
        key1 = 'train dataset'
        key2 = 'loss type'
        key3 = 'tilt'
        key4 = 'latent dim'
        if key1 in line:
            # format has ': ' extra characters after keyword, '\n' after value 
            dataset = line[len(key1) + 2 : -1]
        
        elif key2 in line:
            loss_type = line[len(key2) + 2 : -1]
        
        elif key3 in line:
            tilt = line[len(key3) + 2 : -1]
            if tilt == 'None':
                tilt = None

        elif key4 in line:
            nz = line[len(key4) + 2 : -1]

    return dataset, loss_type, tilt, int(nz)