import matplotlib
matplotlib.use('Agg')
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
from utils.vis import *
from utils.data import *
from model import PoseModel
from transform import *
from dataset import PlanktonDataset
import cPickle as pickle
import numpy as np
import glob
from scipy.spatial.distance import euclidean
from utils.constants import *


model_name = VGG16


def savePredictionCoordinates(coordinates):
    pickle.dump(coordinates, open("best_models/{}/predPose.p".format(model_name), "wb"))
    return 0

def estimateKeyPoints(model, data, use_gpu):
    model.eval()
    inputs, target, coordinates = data['image'], data['target_map'], data['coordinates']
    if use_gpu:
        inputs = Variable(inputs.cuda())
        target = Variable(target.cuda())
    else:
        inputs, target = Variable(inputs), Variable(target)
    outputs = model(inputs)
    poseCoords = []
    for i in range(len(outputs)):
        pred_maps = outputs[i].cpu().data.numpy()
        poseCoords.append(np.stack([np.unravel_index(p.argmax(), p.shape) for p in pred_maps]))
    return poseCoords

def loadModel():
    modelRoot = '/data3/ludi/plankton_wi17/pose/poseprediction_torch/best_models'
    checkpoints = torch.load(modelRoot + '/{}/checkpoints/model_best.pth.tar'.format(model_name))
    gpu_mode = checkpoints.get('gpu_mode', GpuMode.SINGLE)

    model = PoseModel(model_name)
    if gpu_mode == GpuMode.MULTI:
        model = nn.DataParallel(model).cuda()
    elif gpu_mode == GpuMode.SINGLE:
        model = model.cuda(_GPU)

    model.load_state_dict(checkpoints['state_dict'])
    return model

def logEvalStats(metrics):
    assert isinstance(metrics, dict)
    with open('best_models/{}/stats.txt'.format(model_name), "w") as f:
        for cls in metrics:
            print "="*10 + '\n' + cls
            print "Head Distance: {}".format(metrics[cls]['Euclid']['Head Distance'])
            print "Tail Distancce: {}".format (metrics[cls]['Euclid']['Tail Distance'])
            print "Average Distance: {}".format (metrics[cls]['Euclid']['Avg Distance'])
            print "Pose Variability Training Set: {}".format(metrics[cls]['PoseVar'])
            print "KL Divergence: {}".format(metrics[cls]['KLDiv'])

            f.write("="*10 + '\n' + cls + '\n')
            f.write("Head Distance: {}\n".format(metrics[cls]['Euclid']['Head Distance']))
            f.write ("Tail Distancce: {}\n".format (metrics[cls]['Euclid']['Tail Distance']))
            f.write ("Average Distance: {}\n".format (metrics[cls]['Euclid']['Avg Distance']))
            f.write("Pose Variability Training Set: {}".format(metrics[cls]['PoseVar']))
            f.write("KL Divergence: {}".format(metrics[cls]['KLDiv']))
    f.close()

def euclideanDistance(prediction, gtruthHead, gtruthTail):
    headEuclid, tailEuclid = [], []
    head, tail = 0, 1
    nSmpl = len(prediction)
    for i in range(nSmpl):
        headEuclid.append (euclidean (prediction[i][head], gtruthHead[i]))
        tailEuclid.append (euclidean (prediction[i][tail], gtruthTail[i]))
    headEuclid = np.asarray (headEuclid)
    tailEuclid = np.asarray (tailEuclid)
    histData = {'Head Distribution': headEuclid, 'Tail Distribution': tailEuclid}
    avgHeadEuclid = headEuclid.mean ()
    avgTailEuclid = tailEuclid.mean ()
    avgEuclid = np.array ((avgHeadEuclid, avgTailEuclid)).mean ()
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
    img_dir = '/data5/Plankton_wi18/rawcolor_db2/images'
    csv_filename = '/data5/lekevin/plankton/poseprediction/data/data_{}.csv'

    phases = ['train', 'valid', 'test']

    # dataset_mean, dataset_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    normalize = Normalize([0.5, 0.5, 0.5], [1, 1, 1])

    batch_size = 16

    input_size = (384, 384)

    _GPU = 5

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

    # Load datasets
    print 'Loading datasets ...'
    datasets = {x: PlanktonDataset(csv_file=csv_filename.format(x),
                                   img_dir=img_dir,
                                   transform=data_transform[x])
                for x in phases}

    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size,
                                 shuffle=False, num_workers=4)
                   for x in phases}

    dataset_sizes = {x: len(datasets[x]) for x in phases}

    use_gpu = torch.cuda.is_available()

    # Load Model
    print 'Loading model ...'
    model = loadModel()
    predCoordinates = []
    nSmpl = len(dataloaders['test'])

    output_size = get_output_size(model, input_size)

    # Estimate keypoints
    for i,data in enumerate(dataloaders['test']):
        temp = estimateKeyPoints(model, data)
        predCoordinates += temp
        if i%100==0:
            print i,'/',nSmpl
    predCoordinates = [np.fliplr(i) for i in predCoordinates]   # (y,x) --> (x,y)
    predCoordinates = np.asarray(predCoordinates)/float(output_size)           # 48x48 coordinates --> relative head&tail coordinates
    savePredictionCoordinates(predCoordinates)

    # # Temporary
    # temp_predCoordinates = pickle.load(open('predPose.p', "rb"))
    # predCoordinates = np.asarray([np.fliplr(i) for i in temp_predCoordinates])

    # Initialize pose & classes - Test Data
    headX, headY = datasets['test'].data['head_x_rel'], datasets['test'].data['head_y_rel']
    tailX, tailY =  datasets['test'].data['tail_x_rel'], datasets['test'].data['tail_y_rel']
    gtruthHead, gtruthTail, poseTestSet = concatCoordinates(headX, headY, tailX, tailY)
    testTaxLvlDatasets, testspecimenIDs = group_specimen2class(datasets['test'].data['images'])


    # Initialize pose & classes - Train Data
    headX, headY = datasets['train'].data['head_x_rel'], datasets['train'].data['head_y_rel']
    tailX, tailY =  datasets['train'].data['tail_x_rel'], datasets['train'].data['tail_y_rel']
    __, __, poseTrainSet = concatCoordinates(headX, headY, tailX, tailY)
    trainTaxLvlDatasets, trainspecimenIDs = group_specimen2class(datasets['train'].data['images'])


    # Evaluate Model
    print 'Evaluating model (Euclidean, Pose Variability, KL Divergence) ...'

    classMetrics, classMetrics_idx = {}, {}
    for cls in testTaxLvlDatasets['Dataset']:
        # Gather images of each class by their indices from dataset
        trainIdx = [i for i, spc in enumerate(trainspecimenIDs) if spc in trainTaxLvlDatasets['Dataset'][cls]]
        testIdx = [i for i, spc in enumerate(testspecimenIDs) if spc in testTaxLvlDatasets['Dataset'][cls]]

        # Compute metrics
        metrics = {}
        metrics['Euclid'] = euclideanDistance (predCoordinates, gtruthHead[testIdx], gtruthTail[testIdx])
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
