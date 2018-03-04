from __future__ import print_function, division

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

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
from model import PoseClassModel
from transform import *
from logger import Logger
from utils.constants import *
from utils.data import eval_euc_dists, eval_class_acc, get_output_size


parser = argparse.ArgumentParser(description='PyTorch CPM Training')
parser.add_argument('-d', '--dataset-id', default=0, type=int, metavar='N',
                    help='dataset id to use')
parser.add_argument('--data', default='/data3/ludi/plankton_wi17/pose/poseprediction_torch/data/',
                    type=str, metavar='DIR', help='path to dataset')
parser.add_argument('--root', default='/data3/ludi/plankton_wi17/pose/poseprediction_torch/records',
                    type=str, metavar='PATH', help='root directory of the records')
parser.add_argument('--img-dir', default='/data5/Plankton_wi18/rawcolor_db2/images',
                    type=str, metavar='PATH', help='path to images')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--model', '-m', metavar='MODEL', default=VGG16, choices=MODELS,
                    help='pretrained model: ' + ' | '.join(MODELS) + ' (default: {})'.format(VGG16))
# parser.add_argument('-g', '--gpu', default=1, type=int, metavar='N',
#                     help='GPU to use')
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
parser.add_argument('--weight-decay', '--wd', default=0., type=float, metavar='WD',
                    help='weight decay (default: 0)')
parser.add_argument('-w', '--pose-loss-weight', default=100., type=float, metavar='W',
                    help='weight of class loss (default: 1)')
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

phases = [TRAIN, TEST]


def main():
    global args
    args = parser.parse_args()

    csv_filename = os.path.join(args.data, 'pose_class/data_{}_%d.csv' % args.dataset_id)

    print('=> loading model...')
    num_class = DatasetWrapper.get_num_class(csv_filename.format(TRAIN))
    print('=>     {} classes in total'.format(num_class))
    model = PoseClassModel(model_name=args.model, num_class=num_class)
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
                for phase in phases}

    trainer = Trainer(datasets, model, optimizer, exp_lr_scheduler)
    trainer.train()


class Trainer(object):
    def __init__(self, datasets, model, optimizer, scheduler):
        self.datasets = datasets
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
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
        self.best_acc = 0

        if args.resume:
            self.load_checkpoint(os.path.join(self.checkpoints_dir, 'checkpoint.pth.tar'))

        cudnn.benchmark = True

        self.start_epoch = args.start_epoch
        self.end_epoch = self.start_epoch + args.epochs
        print('=> start from epoch {}, end at epoch {}'.format(self.start_epoch, self.end_epoch))

        print('=> initializing logger...')
        # self.loss_logger = Logger('loss', self.log_dir, args.resume)
        # self.err_logger = Logger('err', self.log_dir, args.resume)
        self.log_vars = ['loss', 'class_loss', 'pose_loss', 'class_accuracy', 'pose_error']
        self.main_var = 'class_accuracy'
        self.loggers = {log_var: Logger(log_var, self.log_dir, args.resume, phases=self.phases)
                        for log_var in self.log_vars}

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
            self.best_acc = checkpoint['best_acc']
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

            for phase in self.phases:
                if phase == TRAIN:
                    self.scheduler.step()
                    self.model.train(True)
                else:
                    self.model.train(False)

                running_vars = {var: 0 for var in self.log_vars}
                total = 0
                epoch_since = time.time()

                for i, data in enumerate(self.datasets[phase].dataloader):
                    inputs, target_class, target_map, coordinates = \
                        data['image'], data['class_index'], data['target_map'], data['coordinates']
                    inputs, target_map, target_class = \
                        self.to_variable(inputs), self.to_variable(target_map), self.to_variable(target_class)

                    self.optimizer.zero_grad()

                    outputs_class, outputs_pose = self.model(inputs)
                    loss_class = self.cross_entropy_loss(outputs_class, target_class)
                    loss_pose = self.mse_loss(outputs_pose, target_map) * args.pose_loss_weight
                    loss = loss_class + loss_pose

                    class_acc = eval_class_acc(outputs_class, target_class)
                    pose_err = eval_euc_dists(outputs_pose.cpu().data.numpy(), coordinates.numpy())

                    vars = {'loss': loss.data[0],
                            'class_loss': loss_class.data[0],
                            'pose_loss': loss_pose.data[0],
                            'class_accuracy': class_acc,
                            'pose_error': pose_err['average']}

                    running_vars = {var: running_vars[var] + vars[var] * inputs.size(0) for var in self.log_vars}

                    if phase == TRAIN:
                        loss.backward()
                        self.optimizer.step()

                        for var in self.log_vars:
                            self.loggers[var].add_record(phase, vars[var])

                    total += inputs.size(0)
                    eta = (time.time() - epoch_since) / total * (len(self.datasets[phase]) - total)

                    term_log = ', '.join(['{}: {:.4f}'.format(var, running_vars[var] / total) for var in self.log_vars])
                    print('{} {}/{} ({:.0f}%), {}, ETA: {:.0f}s     \r'
                          .format('Training' if phase == TRAIN else 'validating', total, len(self.datasets[phase]),
                                  100.0 * total / len(self.datasets[phase]), term_log, eta), end='')

                epoch_vars = {var: running_vars[var] / len(self.datasets[phase]) for var in self.log_vars}

                if phase == VALID:
                    for var in self.log_vars:
                        self.loggers[var].add_record(phase, epoch_vars[var])
                    is_best = epoch_vars[self.main_var] < self.best_acc
                    if is_best:
                        self.best_acc = epoch_vars[self.main_var]
                        self.best_model_wts = copy.deepcopy(self.model.state_dict())

                print()
                term_log = ', '.join(['{}: {:.4f}'.format(var, epoch_vars[var]) for var in self.log_vars])
                print('{} {} Time Elapsed: {:.0f}s'
                      .format(phase, term_log, time.time() - epoch_since))
                print()

            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_acc': self.best_acc,
                'optimizer': self.optimizer.state_dict(),
                'gpu_mode': self.gpu_mode
            }, is_best)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}h, {:.0f}m'.format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60))

        if VALID in self.phases:
            print('Best val error: {:4f}'.format(self.best_acc))
            self.model.load_state_dict(self.best_model_wts)

        return self.model


if __name__ == '__main__':
    main()
