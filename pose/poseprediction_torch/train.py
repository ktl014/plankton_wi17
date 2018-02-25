from __future__ import print_function, division

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import copy
from transform import *
from dataset import PlanktonDataset
from model import PoseModel
from visualdl import LogWriter
from utils.data import eval_euc_dists
import shutil


def save_checkpoint(state, is_best, filename='model_checkpoints/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_checkpoints/model_best.pth.tar')


def train_model(model, criterion, optimizer, scheduler, logger, num_epoch=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_err = float('inf')
    is_best = False

    for phase in ['train', 'valid']:
        try:
            with logger.mode(phase):
                if phase not in losses:
                    losses[phase] = logger.scalar('scalars/{}_loss'.format(phase))
                if phase not in errs:
                    errs[phase] = logger.scalar('scalars/{}_err'.format(phase))
            cnt = 0
        except:
            pass

    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch, num_epoch))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_err = 0.0

            epoch_since = time.time()
            total = 0

            for i, data in enumerate(dataloaders[phase]):
                inputs, target, coordinates = data['image'], data['target_map'], data['coordinates']

                if use_gpu:
                    inputs = Variable(inputs.cuda(_GPU))
                    target = Variable(target.cuda(_GPU))
                else:
                    inputs, target = Variable(inputs), Variable(target)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, target)
                err = eval_euc_dists(outputs.cpu().data.numpy(), coordinates.numpy())

                running_loss += loss.data[0] * inputs.size(0)
                running_err += err['average'] * inputs.size(0)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                    losses[phase].add_record(cnt, loss.data[0])
                    errs[phase].add_record(cnt, err['average'])
                    cnt += 1

                eta = (time.time() - epoch_since) / (i + 1) * (n_per_epoch - i - 1)
                total += inputs.size(0)

                print('{} {}/{} ({:.0f}%), Loss: {:.4f}, Error: {:.4f}, ETA: {:.0f}s     \r'
                      .format('Training' if phase == 'train' else 'validating',
                              total, len(datasets[phase]), 100.0 * (i + 1) / n_per_epoch,
                              running_loss / total, running_err / total, eta), end='')

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_err = running_err / dataset_sizes[phase]

            if phase == 'valid':
                losses[phase].add_record(epoch, epoch_loss)
                errs[phase].add_record(epoch, epoch_err)
                if epoch_err < best_err:
                    is_best = True
                    best_err = epoch_err
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()
            print('{} Loss: {:.4f} Error: {:.4f}'.format(phase, epoch_loss, epoch_err))
            print()

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_err': best_err,
            'optimizer': optimizer.state_dict(),
        }, is_best)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}, {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val error: {:4f}'.format(best_err))

    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    log_dir = "./log"
    img_dir = '/data5/Plankton_wi18/rawcolor_db/images'
    csv_filename = 'data/data_{}.csv'

    phases = ['train', 'valid', 'test']

    # dataset_mean, dataset_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    normalize = Normalize([0.5, 0.5, 0.5], [1, 1, 1])

    batch_size = 16

    input_size = (384, 384)

    _GPU = 3

    data_transform = {
        'train': transforms.Compose([
            Rescale(input_size),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor(),
            normalize
        ]),
        'valid': transforms.Compose([
            Rescale(input_size),
            ToTensor(),
            normalize
        ]),
        'test': transforms.Compose([
            Rescale(input_size),
            ToTensor(),
            normalize
        ])
    }

    datasets = {x: PlanktonDataset(csv_file=csv_filename.format(x),
                                   img_dir=img_dir,
                                   transform=data_transform[x])
                for x in phases}

    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size,
                                 shuffle=True, num_workers=4)
                   for x in phases}

    dataset_sizes = {x: len(datasets[x]) for x in phases}

    use_gpu = torch.cuda.is_available()

    n_per_epoch = int(np.ceil(len(datasets['train']) / batch_size))

    losses, errs = {}, {}

    print('initializing......')

    model = PoseModel().cuda(_GPU)

    criterion = nn.MSELoss()

    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), weight_decay=0.0005)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    logger = LogWriter(log_dir, sync_cycle=10)

    print('done.')
    print('-' * 10)

    model = train_model(model, criterion, optimizer, exp_lr_scheduler, logger, num_epoch=5)
