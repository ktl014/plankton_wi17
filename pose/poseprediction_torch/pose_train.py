from __future__ import print_function, division

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import time
import copy
import argparse
import shutil
import datetime

from dataset import DatasetWrapper
from model import PoseModel
from transform import *
from logger import Logger
from utils.constants import *
from utils.data import eval_euc_dists, get_output_size


parser = argparse.ArgumentParser(description='PyTorch CPM Training')
parser.add_argument('-d', '--dataset-id', default=0, type=int, metavar='N',
                    help='dataset id to use')
parser.add_argument('--data', default='/data5/lekevin/plankton/poseprediction/poseprediction_torch/data/3',
                    type=str, metavar='DIR', help='path to dataset')
parser.add_argument('--root', default='/data6/zzuberi/plankton_wi17/pose/poseprediction_torch/records/',
                    type=str, metavar='PATH', help='root directory of the records')
parser.add_argument('--img-dir', default='/data5/Plankton_wi18/rawcolor_db2/images',
                    type=str, metavar='PATH', help='path to images')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--model', '-m', metavar='MODEL', default=RESNET50, choices=MODELS,
                    help='pretrained model: ' + ' | '.join(MODELS) + ' (default: {})'.format(RESNET50))
parser.add_argument('-g', '--gpu', default=1, type=int, metavar='N',
                    help='GPU to use')
parser.add_argument('--epochs', default=25, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N',
                    help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0., type=float, metavar='W',
                    help='weight decay (default: 0)')
parser.add_argument('--amp', default=1., type=float, metavar='AMPLITUDE',
                    help='amplitude of the gaussian belief map (default: 1)')
parser.add_argument('--std', default=3., type=float, metavar='STD',
                    help='std of the gaussian belief map (default: 3)')
parser.add_argument('-i', '--input-size', default=384, type=int, metavar='N',
                    help='input size of the network (default: 384)')
parser.add_argument('--lr-step-size', default=15, type=int, metavar='N',
                    help='the step size of learning rate scheduler (default: 15)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

PHASES = [TRAIN, TEST]

def main():
    global args
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    csv_filename = os.path.join(args.data, 'data_{}_%d.csv' % args.dataset_id)

    print('=> loading model...')
    model = PoseModel(model_name=args.model)
    criterion = nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=0.1)
    print('=> done!')

    datasets = {phase: DatasetWrapper(phase,
                                      csv_filename=csv_filename.format(phase),
                                      img_dir=args.img_dir,
                                      input_size=(args.input_size, args.input_size),
                                      output_size=get_output_size(model, args.input_size),
                                      batch_size=args.batch_size,
                                      amp=args.amp,
                                      std=args.std)
                for phase in PHASES}

    trainer = Trainer(datasets, model, criterion, optimizer, exp_lr_scheduler)
    trainer.train()


class Trainer(object):
    def __init__(self, datasets, model, criterion, optimizer, scheduler):
        self.datasets = datasets
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.phases = [TRAIN, VALID] if args.evaluate else [TRAIN]
        self.root = self.get_root_dir()
        self.log_dir = os.path.join(self.root, 'log')
        self.checkpoints_dir = os.path.join(self.root, 'checkpoints')

        self.gpu_mode = GpuMode.CPU if not torch.cuda.is_available() \
            else GpuMode.MULTI if os.environ.get("CUDA_VISIBLE_DEVICES", '').count(',') > 0 \
            else GpuMode.SINGLE
        print('=> {} GPU mode, using GPU: {}'.format(self.gpu_mode, os.environ.get("CUDA_VISIBLE_DEVICES", '')))

        self.to_cuda()

        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.best_err = float('inf')

        if args.resume:
            self.load_checkpoint(os.path.join(self.checkpoints_dir, 'checkpoint.pth.tar'))

        cudnn.benchmark = True

        self.start_epoch = args.start_epoch
        self.end_epoch = self.start_epoch + args.epochs
        print('=> start from epoch {}, end at epoch {}'.format(self.start_epoch, self.end_epoch))

        print('=> initializing logger...')
        self.loss_logger = Logger('loss', self.log_dir, args.resume)
        self.err_logger = Logger('err', self.log_dir, args.resume)
        Logger.write_meta(self.root, args)
        print('=> done!')

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if not os.path.isdir(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        filepath = os.path.join(self.checkpoints_dir, filename)
        torch.save(state, filepath)
        if is_best:
            print('=> best model so far, saving...')
            shutil.copyfile(filepath, os.path.join(self.checkpoints_dir, 'model_best.pth.tar'))
            print('=> done!')

    def load_checkpoint(self, filename):
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            args.start_epoch = checkpoint['epoch']
            self.best_err = checkpoint['best_err']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

    @staticmethod
    def get_root_dir():
        if args.resume:
            print('=> resume from {}'.format(args.resume))
            return args.resume
        else:
            while True:
                now = datetime.datetime.now()
                root = os.path.join(args.root, args.model, now.strftime('%Y-%m-%d_%H:%M:%S'))
                if not os.path.isdir(root):
                    os.makedirs(root)
                    print('=> record directory created at {}'.format(root))
                    return root

    def to_cuda(self):
        if self.gpu_mode == GpuMode.MULTI:
            self.model = nn.DataParallel(self.model).cuda()
        elif self.gpu_mode == GpuMode.SINGLE:
            self.model = self.model.cuda(0)

    def to_variable(self, tensor):
        if self.gpu_mode == GpuMode.MULTI:
            return Variable(tensor.cuda())
        elif self.gpu_mode == GpuMode.SINGLE:
            return Variable(tensor.cuda(0))
        else:
            return Variable(tensor)

    def train(self):
        since = time.time()
        is_best = False

        for epoch in range(self.start_epoch, self.end_epoch):
            print('Epoch {}/{}'.format(epoch, self.end_epoch))
            print('-' * 10)

            for phase in [TRAIN]:
                if phase == TRAIN:
                    self.scheduler.step()
                    self.model.train(True)
                else:
                    self.model.train(False)

                running_loss, running_err = 0.0, 0.0
                total = 0
                epoch_since = time.time()

                for i, data in enumerate(self.datasets[phase].dataloader):
                    inputs, target, coordinates = data['image'], data['target_map'], data['coordinates']
                    inputs, target = self.to_variable(inputs), self.to_variable(target)

                    self.optimizer.zero_grad()

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, target)
                    err = eval_euc_dists(outputs.cpu().data.numpy(), coordinates.numpy())

                    running_loss += loss.data.item() * inputs.size(0)
                    running_err += err['average'] * inputs.size(0)

                    if phase == TRAIN:
                        loss.backward()
                        self.optimizer.step()

                        self.loss_logger.add_record(phase, loss.data.item())
                        self.err_logger.add_record(phase, err['average'])

                    total += inputs.size(0)
                    eta = (time.time() - epoch_since) / total * (len(self.datasets[phase]) - total)

                    print('{} {}/{} ({:.0f}%), Loss: {:.4f}, Error: {:.4f}, ETA: {:.0f}s     \r'
                          .format('Training' if phase == TRAIN else 'validating',
                                  total, len(self.datasets[phase]), 100.0 * total / len(self.datasets[phase]),
                                  running_loss / total, running_err / total, eta), end='')

                epoch_loss = running_loss / len(self.datasets[phase])
                epoch_err = running_err / len(self.datasets[phase])

                if phase == VALID:
                    self.loss_logger.add_record(phase, epoch_loss)
                    self.err_logger.add_record(phase, epoch_err)
                    is_best = epoch_err < self.best_err
                    if is_best:
                        self.best_err = epoch_err
                        self.best_model_wts = copy.deepcopy(self.model.state_dict())

                print()
                print('{} Loss: {:.4f} Error: {:.4f} Time Elapsed: {:.0f}s'
                      .format(phase, epoch_loss, epoch_err, time.time() - epoch_since))
                print()
            
            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_err': self.best_err,
                'optimizer': self.optimizer.state_dict(),
                'gpu_mode': self.gpu_mode
            }, is_best)
            if VALID not in self.phases and epoch % 5 == 0:
                self.save_checkpoint(self.model.state_dict(), False, filename='checkpoint-{}.pth.tar'.format(epoch))

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}h, {:.0f}m'.format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60))
        if phase == VALID:
            print('Best val error: {:4f}'.format(self.best_err))

        self.model.load_state_dict(self.best_model_wts)
        return self.model


if __name__ == '__main__':
    main()
