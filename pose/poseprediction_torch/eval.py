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
from accuracy import Accuracy


def savePredictionCoordinates(coordinates):
    pickle.dump(coordinates, open("predPose.p", "wb"))
    return 0

def estimateKeyPoints(model, data):
    model.eval()
    inputs, target, coordinates = data['image'], data['target_map'], data['coordinates']
    if use_gpu:
        inputs = Variable(inputs.cuda(_GPU))
        target = Variable(target.cuda(_GPU))
    else:
        inputs, target = Variable(inputs), Variable(target)
    outputs = model(inputs)
    poseCoords = []
    for i in range(len(outputs)):
        pred_maps = outputs[i].cpu().data.numpy()
        poseCoords.append(np.stack([np.unravel_index(p.argmax(), p.shape) for p in pred_maps]))
    return poseCoords

def loadModel():
    modelRoot = '/data3/ludi/plankton_wi17/pose/poseprediction_torch'
    model = PoseModel()
    model = nn.DataParallel(model)
    model = model.cuda()
    checkpoints = torch.load(modelRoot + '/model_checkpoints_2_26/model_best.pth.tar')
    model.load_state_dict(checkpoints['state_dict'])
    return model

def logEvalStats(metrics):
    assert isinstance(metrics, dict)
    with open('stats.txt', "w") as f:
        for cls in metrics:
            print "="*10 + '\n' + cls
            print "Head Distance: {}".format(metrics[cls][0])
            print "Tail Distancce: {}".format (metrics[cls][1])
            print "Average Distance: {}".format (metrics[cls][2])

            f.write("="*10 + '\n' + cls + '\n')
            f.write("Head Distance: {}\n".format(metrics[cls][0]))
            f.write ("Tail Distancce: {}\n".format (metrics[cls][1]))
            f.write ("Average Distance: {}\n".format (metrics[cls][2]))
    f.close()


if __name__ == '__main__':
    img_dir = '/data5/Plankton_wi18/rawcolor_db/images'
    csv_filename = '/data5/lekevin/plankton/poseprediction/data/data_{}.csv'

    phases = ['train', 'valid', 'test']

    # dataset_mean, dataset_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    normalize = Normalize([0.5, 0.5, 0.5], [1, 1, 1])

    batch_size = 16

    input_size = (384, 384)

    _GPU = 1

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

    print 'Loading model ...'
    model = loadModel()
    predCoordinates = []
    nSmpl = len(dataloaders['test'])
    for i,data in enumerate(dataloaders['test']):
        temp = estimateKeyPoints(model, data)
        predCoordinates += temp
        if i%100==0:
            print i,'/',nSmpl
    predCoordinates = np.asarray(predCoordinates)/48.
    savePredictionCoordinates(predCoordinates)

    predCoordinates = pickle.load(open('predPose.p', "rb"))
    # taxLvl Datasets Order --> Specimen, Genus, Family, Order, Dataset
    headX, headY = datasets['test'].data['head_x_rel'], datasets['test'].data['head_y_rel']
    tailX, tailY =  datasets['test'].data['tail_x_rel'], datasets['test'].data['tail_y_rel']
    testTaxLvlDatasets, testspecimenIDs = group_specimen2class(datasets['test'].data['images'])
    trainTaxLvlDatasets, trainspecimenIDs = group_specimen2class(datasets['train'].data['images'])

    specimenSet = testTaxLvlDatasets[4]
    classMetrics = {}
    for cls in specimenSet:
        print "="*10, '\n', cls
        idx = [i for i, spc in enumerate(testspecimenIDs) if spc in specimenSet[cls]]
        # randomPoseCoords = np.asarray([np.random.randint(72, size=(2,2)) for i in range(len(idx))])
        accuracyEval = Accuracy (headX[idx], headY[idx], tailX[idx], tailY[idx])
        classMetrics[cls] = accuracyEval.euclideanDistance (predCoordinates)
        classMetrics_idx[cls] = idx

    logEvalStats(classMetrics)
    # classes = classMetrics.keys()
    # order = np.argsort([classMetrics[cls][2] for cls in classes])
    # classes = [classes[i] for i in order]

    # Pose Variability
    specimenSet = trainTaxLvlDatasets[1]
    for cls in specimenSet:
        idx = [i for i, spc in enumerate (trainspecimenIDs) if spc in specimenSet[cls]]
        classMetrics[cls].append (
            pose_variability (headX[idx], headY[idx], tailX[idx], tailY[idx], [trainspecimenIDs[i] for i in idx]))

    avgEuclid = [classMetrics[cls][2] for cls in specimenSet]
    poseVar = [classMetrics[cls][4] for cls in specimenSet]
    plt.scatter(poseVar, avgEuclid, marker='x')
    plt.xlabel('Pose Variability'); plt.ylabel('Normalized Euclidean Distance')
    plt.title('Pose Variability vs Normalized Distance')
    plt.show()
    plt.savefig('poseVar.png')