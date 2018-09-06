from __future__ import print_function, division

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import math
import torch.utils.model_zoo as model_zoo

import time
import copy
import argparse
import shutil
import datetime
import csv
from sklearn.neighbors import KNeighborsClassifier

from dataset import DatasetWrapper
from models.resnettrip import *
from transform import *
from logger import Logger
from losses import TripletLoss
from utils.constants import *
from utils.data import eval_euc_dists, evaulateTriplet, get_output_size


parser = argparse.ArgumentParser(description='PyTorch CPM Training')
parser.add_argument('-d', '--dataset-id', default=0, type=int, metavar='N',
                    help='dataset id to use')
parser.add_argument('--data', default='/data5/lekevin/plankton/poseprediction/poseprediction_torch/data/3_flipped',
                    type=str, metavar='DIR', help='path to dataset')
parser.add_argument('--root', default='/data6/zzuberi/plankton_wi17/pose/poseprediction_torch/records/',
                    type=str, metavar='PATH', help='root directory of the records')
parser.add_argument('--img-dir', default='/data5/Plankton_wi18/rawcolor_db2/images/',
                    type=str, metavar='PATH', help='path to images')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--model', '-m', metavar='MODEL', default=RESNET50, choices=MODELS,
                    help='pretrained model: ' + ' | '.join(MODELS) + ' (default: {})'.format(RESNET50))
parser.add_argument('-g', '--gpu', required=True, type=int, metavar='N',
                     help='GPU to use')
parser.add_argument('--epochs', default=31, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N',
                    help='mini-batch size (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--lr-decay', default=0.1, type=float,
                    metavar='W', help='learning rate decay (default: 0.1)')
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
parser.add_argument('-i', '--input-size', default=224, type=int, metavar='N',
                    help='input size of the network (default: 384)')
parser.add_argument('--lr_step_size', default=30, type=int, metavar='N',
                    help='the step size of learning rate scheduler (default: 30)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--scratch', dest='scratch', action='store_false',
                    help='train model from scratch')
parser.add_argument('--margin', dest='margin', default=1, type=float, metavar='M',
                    help='loss margin')

phases = [TRAIN]


def main():
    global args
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    csv_filename = os.path.join(args.data, 'data_{}_%d.csv' % args.dataset_id)

    print('=> loading model...')
    num_class = DatasetWrapper.get_num_class(csv_filename.format(TRAIN))
    print('=>     {} classes in total'.format(num_class))
    model = resnettrip(pre_trained=args.scratch)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_decay)
    print('=> done!')

    datasets = {TRAIN: DatasetWrapper(TRAIN,
                                      csv_filename=csv_filename.format(TRAIN),
                                      img_dir=args.img_dir,
                                      input_size=(args.input_size, args.input_size),
                                      #output_size=get_output_size(model, args.input_size),
                                      batch_size=args.batch_size,
                                      amp=args.amp,
                                      std=args.std)
               }
    
    if args.evaluate:
        datasets[TEST] = DatasetWrapper(TEST,
                                    csv_filename=csv_filename.format(TEST),
                                    img_dir=args.img_dir,
                                    input_size=(args.input_size, args.input_size),
                                    #output_size=get_output_size(model, args.input_size),
                                    batch_size=args.batch_size,
                                    amp=args.amp,
                                    std=args.std
                                   )
    
    trainer = Trainer(datasets, model, optimizer, exp_lr_scheduler)
    trainer.train()

class Trainer(object):
    def __init__(self, datasets, model, optimizer, scheduler):
        self.datasets = datasets
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = TripletLoss(args.margin)
        self.phases = [TRAIN, TEST] if args.evaluate else [TRAIN]

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
            self.load_checkpoint(os.path.join(self.checkpoints_dir, 'model_best.pth.tar'))

        cudnn.benchmark = True
        
        argdict = vars(args)
        with open(os.path.join(self.root,'args.csv'), 'wb') as args_file:
            writer = csv.writer(args_file)
            for key, value in argdict.items():
                writer.writerow([key,value])

        self.start_epoch = args.start_epoch
        self.end_epoch = args.epochs
        print('=> start from epoch {}, end at epoch {}'.format(self.start_epoch, self.end_epoch))

        print('=> initializing logger...')
        # self.loss_logger = Logger('loss', self.log_dir, args.resume)
        # self.err_logger = Logger('err', self.log_dir, args.resume)
        self.log_vars = ['loss', 'true_accept', 'false_accept', 'accuracy']
        self.main_var = 'accuracy'
        self.loggers = {log_var: Logger(log_var, self.log_dir, args.resume, phases=self.phases)
                        for log_var in self.log_vars}
        
        self.classifier = KNeighborsClassifier()

        Logger.write_meta(self.root, args)
        print('=> done!')

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if not os.path.isdir(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        if is_best:
            print('=> best model so far, saving...')
            filename = 'model_best.pth.tar'
        filepath = os.path.join(self.checkpoints_dir, filename)
        torch.save(state, filepath)

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
            self.model = self.model.cuda()

    def to_variable(self, tensor):
        if self.gpu_mode == GpuMode.MULTI:
            return Variable(tensor.cuda())
        elif self.gpu_mode == GpuMode.SINGLE:
            return Variable(tensor.cuda())
        else:
            return Variable(tensor)

    def train(self):
        since = time.time()
        is_best = False
        
        for x in range(0,self.start_epoch):
            self.scheduler.step()

        for epoch in range(self.start_epoch, self.end_epoch):
            print('Epoch {}/{}'.format(epoch, self.end_epoch))
            print('-' * 10)
            
            X = []
            y  =[]
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
                    inputs = data['views']
                    targets = data['targets']
                    inputs = np.stack(inputs, axis=1)
                    targets = np.stack(targets,axis=1)

                    inputs = torch.from_numpy(inputs)
                    inputs = self.to_variable(inputs)
                    embeddings = self.model(inputs)
                    
                    loss = self.loss(embeddings[0],embeddings[1],embeddings[2])
                    
                    
                    ta,fa = evaulateTriplet(embeddings[0],embeddings[1],embeddings[2])
                    if phase == TRAIN:
                        for x in range(3):
                            accuracy = 0
                            X.extend(embeddings[x].cpu().detach().numpy())
                            y.extend(targets[:,x])
                    elif phase == TEST:
                        for x in range(3):
                            prediction = self.classifier.predict(embeddings[x].cpu().detach().numpy())
                            accuracy = sum(prediction == targets[:,x])
                        
                    vars = {'loss': loss.data.item(),
                            'true_accept': ta.item(),
                            'false_accept': fa.item(),
                            'accuracy': accuracy}
                    
                    
                    running_vars = {var: running_vars[var] + vars[var] for var in self.log_vars}
                    
                    if phase == TRAIN:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        for var in self.log_vars:
                            self.loggers[var].add_record(phase, vars[var])
                    
                    total += targets.shape[0]
                    
                    eta = (time.time() - epoch_since) / total * (len(self.datasets[phase]) - total)

                    term_log = ', '.join(['{}: {:.4f}'.format(var, running_vars[var] / float(total)) for var in self.log_vars[0:3]])
                    term_log = ', '.join([term_log, '{}: {:.4f}'.format(self.main_var, running_vars[self.main_var] / (3*float(total)))])
                    print('{} {}/{} ({:.0f}%), {}, ETA: {:.0f}s     \r'
                          .format('Training' if phase == TRAIN else 'Validating', total, len(self.datasets[phase]),
                                  100.0 * total / len(self.datasets[phase]), term_log, eta), end='')
                
                epoch_vars = {var: running_vars[var] / float(total) for var in self.log_vars[0:3]}
                epoch_vars[self.main_var] = running_vars[self.main_var] / (3*float(total))

                if phase == TEST:
                    for var in self.log_vars:
                        self.loggers[var].add_record(phase, epoch_vars[var])
                    
                    is_best = self.best_acc < epoch_vars[self.main_var]
                    if is_best:
                        self.best_acc = epoch_vars[self.main_var]
                        self.best_model_wts = copy.deepcopy(self.model.state_dict())
                        
                if TEST in self.phases and phase == TRAIN:
                    self.classifier.fit(X,y)
                
                print()
                term_log = ', '.join(['{}: {:.4f}'.format(var, epoch_vars[var]) for var in self.log_vars])
                print('{} {} Time Elapsed: {:.0f}s'
                      .format(phase, term_log, time.time() - epoch_since))
                print()
                
                
            
            state = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_acc': self.best_acc,
                'optimizer': self.optimizer.state_dict(),
                'gpu_mode': self.gpu_mode
            }
            
            self.save_checkpoint(state, is_best)
            if TEST not in self.phases and epoch % 5 == 0:
                self.save_checkpoint(state, False, filename='checkpoint-{}.pth.tar'.format(epoch))

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}h, {:.0f}m'.format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60))

        if TEST in self.phases:
            print('Best val error: {:4f}'.format(self.best_acc))
            self.model.load_state_dict(self.best_model_wts)

        return self.model


if __name__ == '__main__':
    main()
