from __future__ import print_function, division

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "9"

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import time

from dataset import DatasetWrapper
from model import PoseClassModel
from utils.constants import *
from utils.data import get_output_size, eval_class_acc, eval_euc_dists, get_pred_classes, get_pred_coordinates
from logger import Logger


class Evaluator(object):
    def __init__(self, model_class):
        self.model_class = model_class
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.phase = TEST
        cudnn.benchmark = True
        self.log_vars = ['loss', 'class_loss', 'pose_loss', 'class_accuracy', 'pose_error']

        # preload
        self.model_name, self.num_class = RESNET50, 19
        print('=> loading model...  ', end='')
        self.model = self.model_class(model_name=self.model_name, num_class=self.num_class)
        print('done')

    def set_root(self, root):
        """
        set the root of the model checkpoints to evaluate
        :param root: eg: '/data5/ludi/plankton_wi17/pose/poseprediction_torch/records/resnet50/0/'
        :return: no return
        """
        self.args = Logger.read_meta(os.path.join(root, 'meta.txt'))
        csv_filename = os.path.join(self.args.data, 'pose_class/data_{}_{}.csv'.format(self.phase, self.args.dataset_id))
        num_class = DatasetWrapper.get_num_class(csv_filename.format(self.phase))

        if self.args.model != self.model_name or num_class != self.num_class:
            print('=> reloading model, model name {} -> {}, number of classes: {} -> {}'
                  .format(self.model_name, self.args.model, self.num_class, num_class))
            self.model = self.model_class(model_name=self.model_name, num_class=self.num_class)
            print('=> done')
        self.model_name, self.num_class = self.args.model, num_class

        self.dataset = DatasetWrapper(self.phase,
                                      csv_filename=csv_filename,
                                      img_dir=self.args.img_dir,
                                      input_size=(self.args.input_size, self.args.input_size),
                                      output_size=get_output_size(self.model.cpu(), self.args.input_size),
                                      batch_size=16,
                                      amp=self.args.amp,
                                      std=self.args.std)
        self.root = root

    def to_cuda(self):
        print('=> transferring model to GPU...  ', end='')
        if self.gpu_mode == GpuMode.MULTI:
            self.model = nn.DataParallel(self.model).cuda()
        elif self.gpu_mode == GpuMode.SINGLE:
            self.model = self.model.cuda(0)
        print('done')

    def load_checkpoint(self, filename):
        filepath = os.path.join(self.root, 'checkpoints', filename)
        if os.path.isfile(filepath):
            print("=> Loading checkpoint '{}'".format(filepath))
            checkpoints = torch.load(filepath)

            self.gpu_mode = checkpoints.get('gpu_mode', GpuMode.SINGLE)
            if self.gpu_mode == GpuMode.MULTI:
                self.model = nn.DataParallel(self.model)
            elif self.gpu_mode == GpuMode.SINGLE:
                self.model = self.model
                self.model.load_state_dict(checkpoints['state_dict'])
            print("=> Loaded checkpoint")
        else:
            print("=> no checkpoint found at '{}'".format(filepath))

    def get_checkpoints(self):
        """
        iterate through all the checkpoints under self.roots
        :return: generator of checkpoints filename
        """
        checkpoint_dir = os.path.join(self.root, 'checkpoints')
        checkpoints = os.listdir(checkpoint_dir)
        checkpoints.remove('checkpoint.pth.tar')
        checkpoints = sorted(checkpoints, key=lambda fn: int(fn[11:].split('.')[0]), reverse=True)
        for checkpoint in checkpoints:
            yield checkpoint

    def to_variable(self, tensor):
        if self.gpu_mode == GpuMode.MULTI:
            return Variable(tensor.cuda())
        elif self.gpu_mode == GpuMode.SINGLE:
            return Variable(tensor.cuda(0))
        else:
            return Variable(tensor)

    def generator(self, checkpoint):
        self.load_checkpoint(checkpoint)
        self.to_cuda()
        self.model.train(False)

        for i, data in enumerate(self.dataset.dataloader):
            inputs, target_class, target_map, coordinates = \
                data['image'], data['class_index'], data['target_map'], data['coordinates']
            inputs, target_map, target_class = \
                self.to_variable(inputs), self.to_variable(target_map), self.to_variable(target_class)

            outputs_class, outputs_pose = self.model(inputs)
            loss_class = self.cross_entropy_loss(outputs_class, target_class)
            loss_pose = self.mse_loss(outputs_pose, target_map) * self.args.pose_loss_weight
            loss = loss_class + loss_pose

            # TODO: complete these two functions
            pred_coordinates = get_pred_coordinates(outputs_pose)
            pred_classes = get_pred_classes(outputs_class)

            # TODO: add/remove variables to record
            yield {
                'pred_coordinates': pred_coordinates,
                'pred_classes': pred_classes,
                'gt_coordinates': coordinates,
                'gt_classes': target_class,
                'loss': loss,
                'loss_pose': loss_pose,
                'loss_class': loss_class,
                'outputs_class': outputs_class,
                'outputs_pose': outputs_pose,
                'inputs': inputs
            }

    def evaluate_checkpoint(self, checkpoint):
        running_vars = {var: 0 for var in self.log_vars}
        total = 0
        epoch_since = time.time()

        for i, data in enumerate(self.generator(checkpoint)):
            inputs = data['inputs']
            outputs_class, outputs_pose = data['outputs_class'], data['outputs_pose']
            target_class, coordinates = data['gt_classes'], data['gt_coordinates']
            loss, loss_class, loss_pose = data['loss'], data['loss_class'], data['loss_pose']

            class_acc = eval_class_acc(outputs_class, target_class)
            pose_err = eval_euc_dists(outputs_pose.cpu().data.numpy(), coordinates.numpy())

            vars = {'loss': loss.data[0],
                    'class_loss': loss_class.data[0],
                    'pose_loss': loss_pose.data[0],
                    'class_accuracy': class_acc,
                    'pose_error': pose_err['average']}

            running_vars = {var: running_vars[var] + vars[var] * inputs.size(0) for var in self.log_vars}
            total += inputs.size(0)
            eta = (time.time() - epoch_since) / total * (len(self.dataset) - total)

            term_log = ', '.join(['{}: {:.4f}'.format(var, running_vars[var] / total) for var in self.log_vars])
            print('{} {}/{} ({:.0f}%), {}, ETA: {:.0f}s     \r'
                  .format('Training' if self.phase == TRAIN else 'validating', total, len(self.dataset),
                          100.0 * total / len(self.dataset), term_log, eta), end='')

        epoch_vars = {var: running_vars[var] / len(self.dataset) for var in self.log_vars}

        print()
        term_log = ', '.join(['{}: {:.4f}'.format(var, epoch_vars[var]) for var in self.log_vars])
        print('{} {} Time Elapsed: {:.0f}s'
              .format(self.phase, term_log, time.time() - epoch_since))
        print()


if __name__ == '__main__':
    evaluator = Evaluator(PoseClassModel)

    """ EXAMPLE FOR ITERATOR"""
    for dataset_id in range(10):
        root = '/data5/ludi/plankton_wi17/pose/poseprediction_torch/records/resnet50/{}/'.format(dataset_id)
        evaluator.set_root(root)
        for i, checkpoint in enumerate(evaluator.get_checkpoints()):
            for data in evaluator.generator(checkpoint):
                inputs = data['input']
                outputs_class, outputs_pose = data['outputs_class'], data['outputs_pose']
                gt_classes, gt_coordinates = data['gt_classes'], data['gt_coordinates']
                pred_classes, pred_coordinates = data['pred_classes'], data['pred_coordinates']
                loss, loss_class, loss_pose = data['loss'], data['loss_class'], data['loss_pose']
                """ Do things """

    """ EXAMPLE FOR EVALUATION """
    # for dataset_id in range(10):
    #     root = '/data5/ludi/plankton_wi17/pose/poseprediction_torch/records/resnet50/{}/'.format(dataset_id)
    #     evaluator.set_root(root)
    #     for i, checkpoint in enumerate(evaluator.get_checkpoints()):
    #         evaluator.evaluate_checkpoint(checkpoint)
    #         if i >= 0:
    #             break
