from __future__ import print_function, division
import matplotlib
matplotlib.use('Agg')
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import time
import cPickle as pickle
import pandas as pd
import argparse
from sklearn.neighbors import KNeighborsClassifier

from mvdataset import DatasetWrapper
from models.resnettrip import *
from utils.constants import *
from utils.data import *
from logger import Logger
from utils.vis import *

DEBUG = False

parser = argparse.ArgumentParser(description='PyTorch CPM Evaluation')
parser.add_argument('-g', '--gpu', required=True, type=int, metavar='N',
                     help='GPU to use')

class Evaluator(object):
    def __init__(self, model):
        self.model = model
        self.phases = [TRAIN, TEST]
        cudnn.benchmark = True
        self.log_vars = ['true_accept', 'false_accept', 'accuracy']
        self.running_vars = {var:[] for var in self.log_vars}
        self.datasetIDs = [9]
        self.datasetMetric = {id:{} for id in self.datasetIDs}
        self.classifier = KNeighborsClassifier()

    def set_root(self, root):
        """
        set the root of the model checkpoints to evaluate
        :param root: eg: '/data5/ludi/plankton_wi17/pose/poseprediction_torch/records/resnet50/0/'
        :return: no return
        """
        self.args = Logger.read_meta(os.path.join(root, 'meta.txt'))
        csv_filename = os.path.join(self.args.data, 'data_{}_%d.csv'% self.args.dataset_id)
        self.num_class  = DatasetWrapper.get_num_class(csv_filename.format(TEST))
        
        self.dataset = {TRAIN: DatasetWrapper(TRAIN,
                                              csv_filename=csv_filename.format(TRAIN),
                                              img_dir=self.args.img_dir,
                                              input_size=(self.args.input_size, self.args.input_size),
                                              batch_size=self.args.batch_size,
                                              amp=self.args.amp,
                                              std=self.args.std,
                                              views=self.args.trainviews
                                             )
                       }
        
        
        self.dataset[TEST] = DatasetWrapper(TEST,
                                            csv_filename=csv_filename.format(TEST),
                                            img_dir=self.args.img_dir,
                                            input_size=(self.args.input_size, self.args.input_size),
                                            batch_size=self.args.batch_size,
                                            amp=self.args.amp,
                                            std=self.args.std,
                                            views = self.args.testviews
                                            )

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
        self.level = 'Family'
        _, _, poseXY = self.initialize_posemetrics(phase=TEST)
        specimenSets, specimenIDs = group_specimen2class (self.dataset[TEST].dataset.data['images'], self.level)
        classIdx = {}
        for cls in specimenSets:
            idx = [i for i, spc in enumerate (specimenIDs) if spc in specimenSets[cls]]
            classIdx[cls] = idx
        return specimenSets, specimenIDs, classIdx, poseXY

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
        if 'checkpoint.pth.tar' in checkpoints:
            checkpoints.remove('checkpoint.pth.tar')
            checkpoints.insert(0,'checkpoint.pth.tar')
        checkpoints[1:] = sorted(checkpoints[1:], key=lambda fn: int(fn[11:].split('.')[0]), reverse=True)
        
        for checkpoint in checkpoints:
            yield checkpoint

    def to_variable(self, tensor):
        if self.gpu_mode == GpuMode.MULTI:
            return Variable(tensor.cuda())
        elif self.gpu_mode == GpuMode.SINGLE:
            return Variable(tensor.cuda(0))
        else:
            return Variable(tensor)
        
    def set_classifier(self,checkpoint):
        self.load_checkpoint(checkpoint)
        self.to_cuda()
        self.model.train(False)
        total = 0
        t1 = time.time()
        X = None
        y = []
        phase = TRAIN
        print('=> Training NN classifier... ')
        print()
        with torch.no_grad():
            for i, data in enumerate(self.dataset[phase].dataloader):
                inputs = data['views']
                targets = data['targets']
                
                part = inputs[0]
                part = np.stack(part,axis=0)
                part = torch.from_numpy(part)
                part = self.to_variable(part)
                embedding = self.model(part)
                
                embedding = np.array([em.view(-1).cpu().numpy() for em in embedding])
                
                if X is None:
                    X = embedding
                else:
                    X = np.vstack((X,embedding))
                
                y.extend(targets[0].cpu().detach().numpy())
                
                total += targets[0].shape[0]
                eta = (time.time () - t1) / total * (len (self.dataset[phase]) - total)
                print ('=> Extracting Training Embeddings {}/{} ({:.0f}%), ETA: {:.0f}s     \r'
                       .format (total, len (self.dataset[phase]),
                                100.0 * total / len (self.dataset[phase]), eta),end='')
                
        self.classifier.fit(X,y)
        print()
        X = None
        y = None
                        
    def generator(self):
        phase = TEST
        t1 = time.time()
        total = 0
        with torch.no_grad():
            for i, data in enumerate(self.dataset[phase].dataloader):
                inputs = data['views']
                targets = data['targets']
                
                inputs = np.stack(inputs, axis=0)
                inputs = torch.from_numpy(inputs)
                inputs = self.to_variable(inputs)
                embeddings = self.model(inputs)
                embeddings = np.array([em.view(-1).cpu().numpy() for em in embeddings])
                prediction = self.classifier.predict(embeddings)
                
                pred_classes = []
                pred_classes.extend(prediction)
                
                total += targets.shape[0]
                eta = (time.time () - t1) / total * (len (self.dataset[phase]) - total)
                print ('=> Classifiying Test Samples {}/{} ({:.0f}%), ETA: {:.0f}s     \r'
                       .format (total, len (self.dataset[phase]),
                                100.0 * total / len (self.dataset[phase]), eta),end='')
                yield {
                    'pred_classes': pred_classes,
                    'gt_classes': targets,
                    'view_ids': views
                }

    def save_predictions(self, pred, gtruth, view_ids, checkpoint, filename='predictions'):
        selected_features = self.dataset[TEST].dataset.data[[
            'images',
            'specimen_id']]
        predictions_dataframe = selected_features.copy()
        # Save predictions
        filepath = os.path.join(self.results_dir, filename)
        print ('=> Saved predictions to {}'.format(filepath))
        predictions_dataframe['predictions'] = pd.Series(pred)
        predictions_dataframe['gtruth'] = pd.Series(gtruth)
        predictions_dataframe.to_csv(filepath + '_' + checkpoint[:-7] + 'csv')

    def score_classification(self, pred, gtruth, view_id, checkpoint):
        # Assert instance here
        if not os.path.isdir(self.results_dir):
            print('=> results directory created at {}'.format(self.results_dir))
            os.makedirs(self.results_dir)

        predCls = invert_batchgrouping(pred)
        gtCls = invert_batchgrouping(gtruth)
        self.save_predictions(predCls, gtCls, view_id,checkpoint=checkpoint)
        totalAccu = sum(np.equal(predCls,gtCls))/float(len(predCls))*100
        print('=> Accuracy: {:0.3f}'.format(totalAccu))

        vars = {'class_accuracy': totalAccu}

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
#         _, specimenIDs, classIdx, poseXY = self.sort_classes_poses_etc()
        cls = [i.split()[0] for i in sorted(self.dataset[TEST].dataset.classes)]
        
#         showClassificationDistribution(np.diag(cmRate), title='Dataset {} Class Accuracies'.format(self.args.dataset_id))
#         plotPoseVarKLDiv(self.results_dir, class_accuracy, self.datasetIDs,
#                          ylbl = 'Class Accuracy')
        showConfusionMatrix(self.results_dir, cmRate, classes=cls, checkpoint=checkpoint,
                            title='Confusion Matrix (Dataset {})'.format(self.args.dataset_id))
#         plotPoseOrientation(self.results_dir, predCls, gtCls, poseXY,
#                             dict(zip(self.dataset[TEST].dataset.classes, class_accuracy)), classIdx, specimenIDs)

    def score_entiredataset(self):
        print('=> Results over {} randomly sampled test sets'.format(len(self.datasetIDs)))
        classAvgAccu= []
        for i in self.datasetMetric:
            if len(self.datasetMetric[i]) == 0:
                continue
            classAvgAccu += [self.datasetMetric[i][cls]['Accuracy'] for cls in self.datasetMetric[i]]
        classAvgAccu = np.array(classAvgAccu)
        datasetAvgAccu = np.array(self.running_vars['class_accuracy'])

        print('=> Overall Accuracy [Avg:{:0.3f}, Min:{:0.3f}, Max:{:0.3f}, Std:{:0.3f}]'.
              format(datasetAvgAccu.mean(), datasetAvgAccu.min(), datasetAvgAccu.max(), np.std(datasetAvgAccu) ))
        print('=> Class Accuracy [Avg:{:0.3f}, Min:{:0.3f}, Max:{:0.3f}, Std:{:0.3f}]'.
              format(classAvgAccu.mean(), classAvgAccu.min(), classAvgAccu.max(), np.std(classAvgAccu)))

        #plotPoseVarKLDiv(self.results_dir, classAvgAccu, self.datasetIDs, ylbl = 'Class Accuracy')

if __name__ == '__main__':
    global args
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    print('=> loading evaluator...  ', end='')
    evaluator = Evaluator(resnettrip())
    print('done')

    """ DATASET ITERATOR """
    dataset_since = time.time ()
    for dataset_id in evaluator.datasetIDs:
        print('=> Dataset {}'.format(dataset_id))
        root = '/data6/zzuberi/plankton_wi17/pose/poseprediction_torch/records/resnet50/{}/'.format(dataset_id)
        
        print('=> loading datasets... ')
        evaluator.set_root(root)
        print('done')
        gt_classes, pred_classes, views = [], [], []

        """ UNCOMMENT & TAB SECTION BELOW TO EVALUATE EACH CHECKPOINT"""
#         for i, checkpoint in enumerate(evaluator.get_checkpoints()):
        defaultCheckPoint = next(evaluator.get_checkpoints(), 0)
        evaluator.set_classifier(defaultCheckPoint)
        for data in evaluator.generator():
            gt_classes.append(data['gt_classes'])
            pred_classes.append(data['pred_classes'])
            views.append(data['view_ids'])

#         with open('/data6/zzuberi/plankton_wi17/pose/poseprediction_torch/records/eval_debug/gtCls.p',"r") as f1:
#             gt_classes = pickle.load(f1)
#         with open('/data6/zzuberi/plankton_wi17/pose/poseprediction_torch/records/eval_debug/predCls.p',"r") as f2:
#             pred_classes = pickle.load(f2)

#         if DEBUG:
#             root = '/data5/lekevin/plankton/poseprediction/poseprediction_torch/'
#             gt_classes = pickle.load(open(root + 'tmp/gtCls.p', "rb"))
#             gt_classes = [gt_classes[i].data for i, idx in enumerate(gt_classes)]
#             pred_classes = pickle.load(open(root + 'tmp/predCls.p', "rb"))

        print()
        print('=> Evaluation Results')
        evaluator.score_classification(pred=pred_classes, gtruth=gt_classes, view_id=views,checkpoint=defaultCheckPoint)
        print ()
        print ('Time Elapsed: {:.0f}s'
               .format (time.time () - dataset_since))
        print ()
        #evaluator.compute_posevariabilityDataset()
#     evaluator.score_entiredataset()


