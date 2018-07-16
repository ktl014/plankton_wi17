import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

import cPickle as pickle
import numpy as np
import glob
from scipy.spatial.distance import euclidean
import argparse

from dataset import DatasetWrapper
from utils.vis import *
from utils.data import *
from model import PoseModel
from transform import *
from dataset import PlanktonDataset
from utils.constants import *


model_name = RESNET50

parser = argparse.ArgumentParser(description='PyTorch CPM Evaluation')
parser.add_argument('-g', '--gpu', required=True, type=int, metavar='N',
                     help='GPU to use')
parser.add_argument('-d', '--dataset-id', default=0, type=int, metavar='N',
                    help='dataset id to use')
parser.add_argument('--root', default='/data6/zzuberi/plankton_wi17/pose/poseprediction_torch/records/',
                    type=str, metavar='PATH', help='root directory of the records')
parser.add_argument('--img-dir', default='/data5/Plankton_wi18/rawcolor_db2/images',
                    type=str, metavar='PATH', help='path to images')
parser.add_argument('--model', '-m', metavar='MODEL', default=RESNET50, choices=MODELS,
                    help='pretrained model: ' + ' | '.join(MODELS) + ' (default: {})'.format(RESNET50))
parser.add_argument('--data', default='/data5/lekevin/plankton/poseprediction/poseprediction_torch/data/3',
                    type=str, metavar='DIR', help='path to dataset')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N',
                    help='mini-batch size (default: 16)')
parser.add_argument('-i', '--input-size', default=384, type=int, metavar='N',
                    help='input size of the network (default: 384)')
parser.add_argument('--amp', default=1., type=float, metavar='AMPLITUDE',
                    help='amplitude of the gaussian belief map (default: 1)')
parser.add_argument('--std', default=3., type=float, metavar='STD',
                    help='std of the gaussian belief map (default: 3)')

def get_checkpoints():
        """
        iterate through all the checkpoints under self.roots
        :return: generator of checkpoints filename
        """
        checkpoint_dir = os.path.join(args.root, 'checkpoints')
        checkpoints = os.listdir(checkpoint_dir)
        if 'checkpoint.pth.tar' in checkpoints:
            checkpoints.remove('checkpoint.pth.tar')
        checkpoints = sorted(checkpoints, key=lambda fn: int(fn[11:].split('.')[0]), reverse=True)
        for checkpoint in checkpoints:
            yield checkpoint
            
def savePredictionCoordinates(coordinates):
    print 'Saving prediction in ' + os.path.join(args.root,'predPose.p')
    pickle.dump(coordinates, open(os.path.join(args.root,'predPose.p'), "w+b"))
    return 0

def estimateKeyPoints(model, data):
    model.eval()
    inputs, target, coordinates = data['image'], data['target_map'], data['coordinates']
    if use_gpu:
        inputs = Variable(inputs.cuda(0))
        target = Variable(target.cuda(0))
    else:
        inputs, target = Variable(inputs), Variable(target)
    outputs = model(inputs)
    poseCoords = []
    for i in range(len(outputs)):
        pred_maps = outputs[i].cpu().data.numpy()
        poseCoords.append(np.stack([np.unravel_index(p.argmax(), p.shape) for p in pred_maps]))
    return poseCoords

def loadModel():
    modelRoot = args.root
    model = PoseModel(model_name)
    data_set = 0
    # model = nn.DataParallel(model)  #TODO modify for AlexNet
    args.root = os.path.join(args.root, args.model,str(args.dataset_id))
#     defaultCheckPoint = next(get_checkpoints(), 2)
    defaultCheckPoint = 'checkpoint-15.pth.tar'
    print 'Loading checkpoint ' + os.path.join(args.root,'checkpoints/',defaultCheckPoint)
    checkpoints = torch.load(os.path.join(args.root,'checkpoints/',defaultCheckPoint))
    model.load_state_dict(checkpoints['state_dict'])
    model = model.cuda(0)
    return model

def logEvalStats(metrics):
    assert isinstance(metrics, dict)
    print 'Saving stats in ' + os.path.join(args.root,'stats.txt')
    with open(os.path.join(args.root,'stats.txt'), "w") as f:
        for cls in metrics.keys().sort():
            print "="*10 + '\n' + cls
            print "Head Distance: {}".format(metrics[cls]['Euclid']['Head Distance'])
            print "Tail Distancce: {}".format (metrics[cls]['Euclid']['Tail Distance'])
            print "Average Distance: {}".format (metrics[cls]['Euclid']['Avg Distance'])
            print "Pose Variability Training Set: {}".format(metrics[cls]['PoseVar'])
            print "KL Divergence: {}".format(metrics[cls]['KLDiv'])

            f.write("="*10 + '\n' + cls + '\n')
            f.write("Head Distance: {}\n".format(metrics[cls]['Euclid']['Head Distance']))
            f.write ("Tail Distance: {}\n".format (metrics[cls]['Euclid']['Tail Distance']))
            f.write ("Average Distance: {}\n".format (metrics[cls]['Euclid']['Avg Distance']))
            f.write("Pose Variability Training Set: {}\n".format(metrics[cls]['PoseVar']))
            f.write("KL Divergence: {}\n".format(metrics[cls]['KLDiv']))
    f.close()

def euclideanDistance(prediction, gtruthHead, gtruthTail):
    headEuclid, tailEuclid = [], []
    head, tail = 0, 1
    nSmpl = len(prediction)
    
    for i in range(nSmpl):
        headEuclid.append(euclidean(prediction[i][head], gtruthHead[i]))
        tailEuclid.append(euclidean(prediction[i][tail], gtruthTail[i]))
    headEuclid = np.asarray(headEuclid)
    tailEuclid = np.asarray(tailEuclid)
    histData = {'Head Distribution': headEuclid, 'Tail Distribution': tailEuclid}
    avgHeadEuclid = headEuclid.mean()
    avgTailEuclid = tailEuclid.mean()
    avgEuclid = np.array((avgHeadEuclid, avgTailEuclid)).mean()
    return {'Head Distance':avgHeadEuclid,
            'Tail Distance':avgTailEuclid,
            'Avg Distance':avgEuclid,
            'Distribution':histData}

def concatCoordinates(headX, headY, tailX, tailY):
    assert headX.shape == headY.shape and tailX.shape == tailY.shape
    headXY = np.column_stack ((headX, headY))
    tailXY = np.column_stack ((tailX, tailY))
    poseXY = np.stack([headX - tailX, headY - tailY], axis=1)
    return headXY, tailXY, poseXY

if __name__ == '__main__':
    global args
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    csv_filename = os.path.join(args.data, 'data_{}_%d.csv' % args.dataset_id)

    phases = [TRAIN, TEST]
    
    # dataset_mean, dataset_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    normalize = Normalize([0.5, 0.5, 0.5], [1, 1, 1])

    use_gpu = torch.cuda.is_available()

    # Load Model
    print 'Loading model ...'
    model = loadModel()
    
    # Load datasets
    print 'Loading datasets ...'
    datasets = {phase: DatasetWrapper(phase,
                                      csv_filename=csv_filename.format(phase),
                                      img_dir=args.img_dir,
                                      input_size=(args.input_size, args.input_size),
                                      output_size=get_output_size(model, args.input_size),
                                      batch_size=args.batch_size,
                                      amp=args.amp,
                                      std=args.std)
                for phase in phases}
    
    predCoordinates = []
    nSmpl = datasets[TEST].dataset_size

    # Estimate keypoints
    for i,data in enumerate(datasets[TEST].dataloader):
        temp = estimateKeyPoints(model, data)
        predCoordinates += temp
        if i%100==0:
            print i,'/',nSmpl
    predCoordinates = [np.fliplr(i) for i in predCoordinates]   # (y,x) --> (x,y)
    predCoordinates = np.asarray(predCoordinates)/48.           # 48x48 coordinates --> relative head&tail coordinates
    savePredictionCoordinates(predCoordinates)

    # # Debug
#     temp_predCoordinates = pickle.load(open(os.path.join(args.root,'predPose.p'), "rb"))
#     predCoordinates = np.asarray([np.fliplr(i) for i in temp_predCoordinates])
    
    # Initialize pose & classes - Test Data
    headX, headY = datasets[TEST].dataset.data['head_x_rel'], datasets[TEST].dataset.data['head_y_rel']
    tailX, tailY =  datasets[TEST].dataset.data['tail_x_rel'], datasets[TEST].dataset.data['tail_y_rel']
    gtruthHead, gtruthTail, poseTestSet = concatCoordinates(headX, headY, tailX, tailY)
    testTaxLvlDatasets, testspecimenIDs = group_specimen2class(datasets[TEST].dataset.data['images'],FAMILY)


    # Initialize pose & classes - Train Data
    headX, headY = datasets[TRAIN].dataset.data['head_x_rel'], datasets[TRAIN].dataset.data['head_y_rel']
    tailX, tailY =  datasets[TRAIN].dataset.data['tail_x_rel'], datasets[TRAIN].dataset.data['tail_y_rel']
    __, __, poseTrainSet = concatCoordinates(headX, headY, tailX, tailY)
    trainTaxLvlDatasets, trainspecimenIDs = group_specimen2class(datasets[TRAIN].dataset.data['images'],FAMILY)

    # Evaluate Model
    print 'Evaluating model (Euclidean, Pose Variability, KL Divergence) ...'

    classMetrics, classMetrics_idx = {}, {}
    for cls in testTaxLvlDatasets:
        # Gather images of each class by their indices from dataset
        trainIdx = [i for i, spc in enumerate(trainspecimenIDs) if spc in trainTaxLvlDatasets[cls]]
        testIdx = [i for i, spc in enumerate(testspecimenIDs) if spc in testTaxLvlDatasets[cls]]

        # Compute metrics
        metrics = {}
        metrics['Euclid'] = euclideanDistance (predCoordinates[testIdx], gtruthHead[testIdx], gtruthTail[testIdx])
        metrics['PoseVar'] = pose_variability2(poseTrainSet[trainIdx], [trainspecimenIDs[i] for i in trainIdx])
        metrics['KLDiv'] = pose_diff2(poseTrainSet[trainIdx], [trainspecimenIDs[i] for i in trainIdx],
                                           poseTestSet[testIdx], [testspecimenIDs[i] for i in testIdx])
        classMetrics[cls] = metrics
    logEvalStats(classMetrics)

    # 2)Jupyter notebook should just visualize those results by opening predCoordinates and gtruth, then looking at distribution for each class
    # must specify for which experiment to look at predPose for then

    # 3) optional -- Temp solution (create euclidean distance function)
    # Use Metrics class to create the classMetrics dictionary (head euclid, tail euclid, avg euclid, pose variability) given the pose variability and test data

    # showHeadTailDistribution(classMetrics['Dataset'][3])

    # avgEuclid = [classMetrics[cls][2] for cls in specimenSet]
    # poseVar = [classMetrics[cls][4] for cls in specimenSet]
    # plt.scatter(poseVar, avgEuclid, marker='x')
    # plt.xlabel('Pose Variability'); plt.ylabel('Normalized Euclidean Distance')
    # plt.title('Pose Variability vs Normalized Distance')
    # plt.show()
    # plt.savefig('poseVar.png')
