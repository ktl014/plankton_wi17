from __future__ import print_function, division

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import time
import cPickle as pickle
import pandas as pd

from dataset import DatasetWrapper
from model import PoseClassModel
from utils.constants import *
from utils.data import *
from utils.vis import *
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
        self.running_vars = {var: [] for var in self.log_vars}
        self.datasetIDs = [9]
        self.datasetMetric = {id:{} for id in self.datasetIDs}

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
            self.model = self.model_class(model_name=self.args.model, num_class=num_class)
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
        self.results_dir = os.path.join(self.root, 'results')

    def initialize_posemetrics(self, phase):
        headX, headY = self.dataset[phase].dataset.data['head_x_rel'], self.dataset[phase].dataset.data['head_y_rel']
        tailX, tailY = self.dataset[phase].dataset.data['tail_x_rel'], self.dataset[phase].dataset.data['tail_y_rel']
        headXY, tailXY = np.column_stack ((headX, headY)), np.column_stack ((tailX, tailY))
        poseXY = np.stack([headX - tailX, headY - tailY], axis=1)
        return headXY, tailXY, poseXY

    def sort_classes_poses_etc(self):
        if not os.path.isdir(self.results_dir):
            print('=> results directory created at {}'.format(self.results_dir))
            os.makedirs(self.results_dir)

        self.level = 'Genus'
        self.gtruthHead, self.gtruthTail, self.poseXY = self.initialize_posemetrics(phase=TEST)
        self.specimenSets, self.specimenIDs = group_specimen2class (self.dataset[TEST].dataset.data['images'], self.level)
        self.classIdx = {}
        for cls in self.specimenSets:
            idx = [i for i, spc in enumerate (self.specimenIDs) if spc in self.specimenSets[cls]]
            self.classIdx[cls] = idx

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
        total = 0
        t1 = time.time()
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

            total += inputs.size(0)
            eta = (time.time() - t1) / total * (len(self.dataset[TEST]) - total)

            print ('=> Predicting {}/{} ({:.0f}%), ETA: {:.0f}s     \r'
                   .format (total, len (self.dataset[TEST]),
                            100.0 * total / len(self.dataset[TEST]), eta), end='')

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

    def save_predictions(self, pred, gtruth, filename='predictions.csv'):
        selected_features = self.dataset[TEST].dataset.data[[
            'images',
            'specimen_id']]
        predictions_dataframe = selected_features.copy()

        # Save predictions
        filepath = os.path.join(self.results_dir, filename)
        print ('=> Saved predictions to {}'.format(filepath))
        predictions_dataframe['predictions'] = pred
        predictions_dataframe['gtruth'] = gtruth
        predictions_dataframe.to_csv(filepath)

    def score_classification(self, pred, gtruth):
        # Assert instance here
        predCls = invert_batchgrouping(pred)
        gtCls = invert_batchgrouping(gtruth)

        totalAccu = (predCls == gtCls).mean()*100
        print('=> Accuracy: {:0.3f}'.format(totalAccu))

        vars = {'class_accuracy': totalAccu,
                'loss': 0,
                'class_loss': 0,
                'pose_loss':0,
                'pose_error':0}

        self.running_vars = {var: self.running_vars[var] + [vars[var]] for var in self.log_vars}

        cmRawCount = np.zeros((self.num_class, self.num_class))
        clsCount = np.zeros((self.num_class,1))
        cmRate = np.zeros((self.num_class, self.num_class))

        for t,p in zip(gtCls, predCls):
            clsCount[t,0] += 1
            cmRawCount[t,p] += 1

        for i in range(self.num_class):
            cmRate[i,:] = cmRawCount[i,:] / clsCount[i,0] * 100

        class_accuracy = []
        for i,cls in enumerate(sorted(self.dataset[TEST].dataset.classes)):
            self.datasetMetric[self.args.dataset_id][cls]['Accuracy'] = cmRate.diagonal()[i]
            class_accuracy.append(cmRate.diagonal()[i])

        #TODO Insert histogram of class accuracies / confusion matrix
        cls = [i.split()[0] for i in sorted(self.dataset[TEST].dataset.classes)]
        # showClassificationDistribution(cmRate.diagonal, title='Dataset {} Class Accuracies'.format(self.args.dataset_id))
        plotPoseVarKLDiv(self.results_dir, class_accuracy, self.datasetIDs,
                         ylbl = 'Class Accuracy')
        showConfusionMatrix(self.results_dir, cmRate, classes=cls,
                            title='Confusion Matrix (Dataset {})'.format(self.args.dataset_id))
        plotPoseOrientation(self.results_dir, predCls, gtCls, self.poseXY,
                            dict(zip(self.dataset[TEST].dataset.classes, class_accuracy)), self.classIdx, self.specimenIDs)

    def score_poseprediction(self, pred):
        predCoordinates = np.array(invert_batchgrouping(pred))

        for i,cls in enumerate(self.specimenSets):
            testIdx = self.classIdx[cls]
            print(cls, len(predCoordinates[testIdx]), len(self.gtruthTail[testIdx]), len(self.gtruthHead[testIdx]))
            self.datasetMetric[self.args.dataset_id][cls]['Euclid'] = euclideanDistance(predCoordinates[testIdx], self.gtruthHead[testIdx], self.gtruthTail[testIdx])
        avgEuclid = np.array ([self.datasetMetric[self.args.dataset_id][cls]['Euclid']['Avg Distance'] for cls in self.datasetMetric[self.args.dataset_id]])
        print('=> Avg Euclid: {:0.3f}'.format(avgEuclid.mean()))

        #TODO insert histogram of euclid distance per class

    def score_entiredataset(self):
        print('=> Checkpoint saved')
        pickle.dump (self.datasetMetric, open ('utils/tmp/entireDatasetMetrics.p', "wb"))
        print('=> Results over 10 randomly sampled test sets')
        avgAccu, avgEuclid = [], []
        for i in self.datasetMetric:
            if len(self.datasetMetric[i]) == 0:
                continue
            avgAccu += [self.datasetMetric[i][cls]['Accuracy'] for cls in self.datasetMetric[i]]
            avgEuclid += [self.datasetMetric[i][cls]['Euclid']['Avg Distance'] for cls in self.datasetMetric[i]]
        datasetAvgAccu = np.array (self.running_vars['class_accuracy'])
        avgAccu = np.array(avgAccu)
        avgEuclid = np.array(avgEuclid)

        print ('=> Overall Accuracy [Avg:{:0.3f}, Min:{:0.3f}, Max:{:0.3f}, Std:{:0.3f}]'.
               format (datasetAvgAccu.mean (), datasetAvgAccu.min (), datasetAvgAccu.max (), np.std (datasetAvgAccu)))
        print('=> Class Accuracy [Avg:{:0.3f}, Min:{:0.3f}, Max:{:0.3f}, Std:{:0.3f}]'
              .format(avgAccu.mean(), avgAccu.min(), avgAccu.max(), np.std(avgAccu)))
        print ('=> Class Euclidean [Avg:{:0.3f}, Min:{:0.3f}, Max:{:0.3f}, Std:{:0.3f}]'.
               format (avgEuclid.mean (), avgEuclid.min (), avgEuclid.max (), np.std (avgEuclid)))

        plotPoseVarKLDiv(self.results_dir, avgAccu, self.datasetIDs, ylbl = 'Class Accuracy')
        plotPoseVarKLDiv(self.results_dir, avgEuclid, self.datasetIDs, ylbl ='Normalized Distance')

if __name__ == '__main__':
    evaluator = Evaluator(PoseClassModel)

    if DEBUG:
        LEVEL = 'Genus'  # SPECIMEN GENUS FAMILY ORDER DATASET
        NUM_COLS = 8  # Number of columns for
        user = 'Kevin'
        dataset_id = 9
        EXP_TYPE = 'pose_class'

        root = '/data5/lekevin/plankton/poseprediction/poseprediction_torch/records/resnet50/{}/{}/'.format (
            EXP_TYPE, dataset_id)
        evaluator.set_root (root)

        root = '/data5/lekevin/plankton/poseprediction/poseprediction_torch/'
        predCls = pickle.load (open (root + '/tmp/{}/{}/predCls.p'.format (EXP_TYPE, dataset_id), "rb"))
        gtCls = pickle.load (open (root + '/tmp/{}/{}/gtCls.p'.format (EXP_TYPE, dataset_id), "rb"))
        pred_coordinates = pickle.load (open (root + 'tmp/predCoord.p', "rb"))

        evaluator.sort_classes_poses_etc ()
        evaluator.score_classification (pred=predCls, gtruth=gtCls)
        evaluator.score_poseprediction (pred=pred_coordinates)
        exit(0)

    """ DATASET ITERATOR """
    dataset_since = time.time()
    for dataset_id in evaluator.datasetIDs:
        print('=> Dataset {}'.format(dataset_id))
        root = '/data5/lekevin/plankton/poseprediction/poseprediction_torch/records/resnet50/pose_class/{}/'.format(dataset_id)
        evaluator.set_root(root)
        gt_classes, gt_coordinates = [], []
        pred_classes, pred_coordinates = [], []

        """ UNCOMMENT & TAB SECTION BELOW TO EVALUATE EACH CHECKPOINT"""
        # for i, checkpoint in enumerate(evaluator.get_checkpoints()):
        default_checkpoint = next(evaluator.get_checkpoints(), 0)
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

        print()
        print('=> Dataset {} Evaluation Results'.format(dataset_id))
        evaluator.sort_classes_poses_etc()
        #evaluator.save_predictions(pred=pred_classes, gtruth=gt_classes)
        evaluator.score_classification(pred=pred_classes, gtruth=gt_classes)
        evaluator.score_poseprediction(pred = pred_coordinates)
        #evaluator.compute_posevariabilityDataset()
        print ()
        print ('Time Elapsed: {:.0f}s'
               .format (time.time () - dataset_since))
        print ()

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
    # for dataset_id in range(10):
    #     root = '/data5/ludi/plankton_wi17/pose/poseprediction_torch/records/resnet50/{}/'.format(dataset_id)
    #     evaluator.set_root(root)
    #     for i, checkpoint in enumerate(evaluator.get_checkpoints()):
    #         evaluator.evaluate_checkpoint(checkpoint)
    #         if i >= 0:
    #             break
