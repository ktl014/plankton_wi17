from __future__ import print_function, division

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "9"

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import time
import numpy as np
import cPickle as pickle

from dataset import DatasetWrapper
from model import PoseClassModel
from utils.constants import *
from utils.data import *
from logger import Logger

DEBUG = False

class Evaluator(object):
    def __init__(self, model_class):
        self.model_class = model_class
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.phases = [TRAIN, TEST]
        cudnn.benchmark = True
        self.log_vars = ['loss', 'class_loss', 'pose_loss', 'class_accuracy', 'pose_error']
        self.datasetMetric = {datasetID:{} for datasetID in range(10)}

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
        csv_filename = os.path.join(self.args.data, 'pose_class/data_{}_%d.csv'% self.args.dataset_id)
        num_class = DatasetWrapper.get_num_class(csv_filename.format(TEST))

        if self.args.model != self.model_name or num_class != self.num_class:
            print('=> reloading model, model name {} -> {}, number of classes: {} -> {}'
                  .format(self.model_name, self.args.model, self.num_class, num_class))
            self.model = self.model_class(model_name=self.model_name, num_class=self.num_class)
            print('=> done')
        self.model_name, self.num_class = self.args.model, num_class

        self.dataset = {phase: DatasetWrapper(phase,
                                      csv_filename=csv_filename.format(phase),
                                      img_dir=self.args.img_dir,
                                      input_size=(self.args.input_size, self.args.input_size),
                                      output_size=get_output_size(self.model.cpu(), self.args.input_size),
                                      batch_size=16,
                                      amp=self.args.amp,
                                      std=self.args.std) for phase in self.phases}

        self.datasetMetric[self.args.dataset_id] = {cls:{} for cls in self.dataset[TRAIN].dataset.classes}
        self.root = root

    def initialize_posemetrics(self, phase):
        headX, headY = self.dataset[phase].dataset.data['head_x_rel'], self.dataset[phase].dataset.data['head_y_rel']
        tailX, tailY = self.dataset[phase].dataset.data['tail_x_rel'], self.dataset[phase].dataset.data['tail_y_rel']
        headXY, tailXY = np.column_stack ((headX, headY)), np.column_stack ((tailX, tailY))
        poseXY = np.stack([headX - tailX, headY - tailY], axis=1)
        return headXY, tailXY, poseXY

    def compute_posevariabilityDataset(self):
        self.gtruthHead, self.gtruthTail, self.poseTestSet = self.initialize_posemetrics(phase=TEST)
        self.testTaxLvlDatasets, self.testspecimenIDs = group_specimen2class (self.dataset[TEST].dataset.data['images'])
        self.level = 'Genus'

        _, _, poseTrainSet = self.initialize_posemetrics(phase=TRAIN)
        trainTaxLvlDatasets, trainspecimenIDs = group_specimen2class (self.dataset[TRAIN].dataset.data['images'])

        print('=>Computing dataset pose variability metrics')
        for j,cls in enumerate(self.testTaxLvlDatasets[self.level]):
            trainIdx = [i for i, spc in enumerate (trainspecimenIDs) if spc in trainTaxLvlDatasets[self.level][cls]]
            testIdx = [i for i, spc in enumerate (self.testspecimenIDs) if spc in self.testTaxLvlDatasets[self.level][cls]]

            # Compute metrics
            self.clsMetrics[cls]['PoseVar'] = pose_variability2 (poseTrainSet[trainIdx], [trainspecimenIDs[i] for i in trainIdx])
            self.clsMetrics[cls]['KLDiv'] = pose_diff2 (poseTrainSet[trainIdx], [trainspecimenIDs[i] for i in trainIdx],
                                           self.poseTestSet[testIdx], [self.testspecimenIDs[i] for i in testIdx])
            if j%5==0:
                print('=>{}/{}'.format(j, len(self.testTaxLvlDatasets[self.level])))


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

        for i, data in enumerate(self.dataset[TEST].dataloader):
            inputs, target_class, target_map, coordinates = \
                data['image'], data['class_index'], data['target_map'], data['coordinates']
            inputs, target_map, target_class = \
                self.to_variable(inputs), self.to_variable(target_map), self.to_variable(target_class)

            outputs_class, outputs_pose = self.model(inputs)
            loss_class = self.cross_entropy_loss(outputs_class, target_class)
            loss_pose = self.mse_loss(outputs_pose, target_map) * self.args.pose_loss_weight
            loss = loss_class + loss_pose

            pred_coordinates = get_pred_coordinates(outputs_pose)
            pred_classes = get_pred_classes(outputs_class)

            if i%100==0:
                print('=> {}/{}'.format(i, len(self.dataset[TEST].dataloader)))

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
                  .format('Training' if self.phase == TRAIN else 'testing', total, len(self.dataset),
                          100.0 * total / len(self.dataset), term_log, eta), end='')

        epoch_vars = {var: running_vars[var] / len(self.dataset) for var in self.log_vars}

        print()
        term_log = ', '.join(['{}: {:.4f}'.format(var, epoch_vars[var]) for var in self.log_vars])
        print('{} {} Time Elapsed: {:.0f}s'
              .format(self.phase, term_log, time.time() - epoch_since))
        print()

        return epoch_vars

    def score_classification(self, pred, gtruth):
        # Assert instance here
        predCls = np.array(invert_batchgrouping(pred))
        gtCls = np.array(invert_batchgrouping(gtruth))

        totalAccu = (predCls == gtCls).mean()*100
        print('=> Accuracy: {:0.3f}'.format(totalAccu))

        cmRawCount = np.zeros((self.num_class, self.num_class))
        clsCount = np.zeros((self.num_class,1))
        cmRate = np.zeros((self.num_class, self.num_class))

        for t,p in zip(gtCls, predCls):
            clsCount[t,0] += 1
            cmRawCount[t,p] += 1

        for i in range(self.num_class):
            cmRate[i,:] = cmRawCount[i,:] / clsCount[i,0] * 100

        for i,cls in enumerate(self.datasetMetric[self.args.dataset_id]):
            self.datasetMetric[self.args.dataset_id][cls]['Accuracy'] = cmRate.diagonal()[i]

        #TODO Insert histogram of class accuracies / confusion matrix

    def score_poseprediction(self, pred):
        predCoordinates = np.array(invert_batchgrouping(pred))
        self.gtruthHead, self.gtruthTail, _ = self.initialize_posemetrics(phase=TEST)
        self.testTaxLvlDatasets, self.testspecimenIDs = group_specimen2class (self.dataset[TEST].dataset.data['images'])
        self.level = 'Genus'

        for i,cls in enumerate(self.testTaxLvlDatasets[self.level]):
            testIdx = [i for i, spc in enumerate(self.testspecimenIDs) if spc in self.testTaxLvlDatasets[self.level][cls]]
            self.datasetMetric[self.args.dataset_id][cls]['Euclid'] = euclideanDistance(predCoordinates[testIdx], self.gtruthHead[testIdx], self.gtruthTail[testIdx])
        avgEuclid = np.array ([self.datasetMetric[self.args.dataset_id][cls]['Euclid']['Avg Distance'] for cls in self.datasetMetric[self.args.dataset_id]])
        print('=> Avg Euclid: {:0.3f}'.format(avgEuclid.mean()))

        #TODO insert histogram of euclid distance per class

    def score_entiredataset(self):
        print('=> Results over 10 randomly sampled test sets')
        avgAccu, avgEuclid = [], []
        try:
            for i in self.datasetMetric:
                avgAccu += [self.datasetMetric[i][cls]['Accuracy'] for cls in self.datasetMetric[i]]
                avgEuclid += [self.datasetMetric[i][cls]['Euclid']['Avg Distance'] for cls in self.datasetMetric[i]]
            avgAccu = np.array(avgAccu)
            avgEuclid = np.array(avgEuclid)
            print('=> Overall Accuracy [Avg:{:0.3f}, Min:{:0.3f}, Max:{:0.3f}, Std:{:0.3f}]'.format(avgAccu.mean(),
                                                                                avgAccu.min(),
                                                                                avgAccu.max(),
                                                                                np.std(avgAccu)))
            print ('=> Overall Euclidean [Avg:{:0.3f}, Min:{:0.3f}, Max:{:0.3f}, Std:{:0.3f}]'.format (avgEuclid.mean (),
                                                                                  avgEuclid.min (),
                                                                                  avgEuclid.max (),
                                                                                  np.std (avgEuclid)))

            plotPoseVarKLDiv(avgAccu, ylbl = 'Class Accuracy')
            plotPoseVarKLDiv(avgEuclid, ylbl ='Normalized Distance')
        except:
            print("=> ERROR: Results for all 10 datasets are incomplete")

if __name__ == '__main__':
    evaluator = Evaluator(PoseClassModel)

    """ EXAMPLE FOR ITERATOR"""
    # for dataset_id in range(10):
    #     root = '/data5/ludi/plankton_wi17/pose/poseprediction_torch/records/resnet50/{}/'.format(dataset_id)
    #     evaluator.set_root(root)
    #     for i, checkpoint in enumerate(evaluator.get_checkpoints()):
    #         for data in evaluator.generator(checkpoint):
    #             inputs = data['input']
    #             outputs_class, outputs_pose = data['outputs_class'], data['outputs_pose']
    #             gt_classes, gt_coordinates = data['gt_classes'], data['gt_coordinates']
    #             pred_classes, pred_coordinates = data['pred_classes'], data['pred_coordinates']
    #             loss, loss_class, loss_pose = data['loss'], data['loss_class'], data['loss_pose']
    #             """ Do things """

    """ DATASET ITERATOR """
    for dataset_id in range(10):
        print('=> Dataset {}'.format(dataset_id))
        root = '/data5/ludi/plankton_wi17/pose/poseprediction_torch/records/resnet50/{}/'.format(dataset_id)
        evaluator.set_root(root)
        gt_classes, gt_coordinates = [], []
        pred_classes, pred_coordinates = [], []

        """ UNCOMMENT & TAB SECTION BELOW TO EVALUATE EACH CHECKPOINT"""
        # for i, checkpoint in enumerate(evaluator.get_checkpoints()):
        default_checkpoint = '/data5/ludi/plankton_wi17/pose/poseprediction_torch/records/resnet50/0/checkpoints/checkpoint-15.pth.tar'
        for data in evaluator.generator(default_checkpoint):
            inputs = data['inputs']
            outputs_class, outputs_pose = data['outputs_class'], data['outputs_pose']
            # gt_classes, gt_coordinates = data['gt_classes'], data['gt_coordinates']
            # pred_classes, pred_coordinates = data['pred_classes'], data['pred_coordinates']
            loss, loss_class, loss_pose = data['loss'], data['loss_class'], data['loss_pose']

            gt_classes.append(data['gt_classes'].data)
            gt_coordinates.append(data['gt_coordinates'])
            pred_classes.append(data['pred_classes'])
            pred_coordinates.append(data['pred_coordinates'])

        if DEBUG:
            root = '/data5/lekevin/plankton/poseprediction/poseprediction_torch/'
            gt_classes = pickle.load(open(root + 'tmp/gtCls.p', "rb"))
            gt_classes = [gt_classes[i].data for i, idx in enumerate(gt_classes)]
            pred_classes = pickle.load(open(root + 'tmp/predCls.p', "rb"))
            pred_coordinates = pickle.load(open(root + 'tmp/predCoord.p', "rb"))

        print('=> Evaluation Results')
        evaluator.score_classification(pred=pred_classes, gtruth=gt_classes)
        evaluator.score_poseprediction(pred = pred_coordinates)
        #evaluator.compute_posevariabilityDataset()
    evaluator.score_entiredataset()

    """ TEST: Compute Pose Variability """
    # PoseVarMetrics = {}
    # for dataset_id in range (10):
    #     print ('=> Dataset {}'.format (dataset_id))
    #     root = '/data5/ludi/plankton_wi17/pose/poseprediction_torch/records/resnet50/{}/'.format (dataset_id)
    #     evaluator.set_root (root)
    #     evaluator.compute_posevariabilityDataset ()
    #     PoseVarMetrics[dataset_id] = evaluator.clsMetrics
    #     pickle.dump (PoseVarMetrics, open ('tmp/PoseVarMetrics.p', "wb"))

    """ EXAMPLE FOR EVALUATION """
    # accs = []
    # for dataset_id in range(10):
    #     root = '/data5/ludi/plankton_wi17/pose/poseprediction_torch/records/resnet50/{}/'.format(dataset_id)
    #     evaluator.set_root(root)
    #     acc = 0
    #     for i, checkpoint in enumerate(evaluator.get_checkpoints()):
    #         metrics = evaluator.evaluate_checkpoint(checkpoint)
    #         acc = max(acc, metrics['class_accuracy'])
    #         if i >= 1:
    #             break
    #     accs.append(acc)
    # print(accs)
    # print(np.mean(accs))

    root = '/data5/ludi/plankton_wi17/pose/poseprediction_torch/records/resnet50/2018-03-07_02:20:09'
    evaluator.set_root(root)
    evaluator.evaluate_checkpoint('checkpoint-19.pth.tar')
