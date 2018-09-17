import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchvision import utils
import torch
from scipy.spatial.distance import euclidean
import cPickle as pickle
import itertools
from utils.data import specimen_normalization

def show_arrow(image, coordinates, cls):
    if isinstance(image, (np.ndarray, list)):
        image = np.copy(image)
    elif isinstance(image, torch.FloatTensor):
        image = (image.numpy()).transpose((1, 2, 0)).copy()

    if isinstance(coordinates, torch.FloatTensor):
        coordinates = coordinates.numpy()

    height, width = image.shape[:2]
    head = (int(coordinates[0] * width), int(coordinates[1] * height))
    tail = (int(coordinates[2] * width), int(coordinates[3] * height))
    cv2.arrowedLine(image, tail, head, (1., 0., 0.), 3)
    plt.imshow(image)
    plt.axis('off')
    plt.title(cls)
    plt.pause(0.001)

def evalShowArrow(image, predcoordinates, gtruthcoordinates, cls):
    if isinstance(image, (np.ndarray, list)):
        image = np.copy(image)
    elif isinstance(image, torch.FloatTensor):
        image = (image.numpy()).transpose((1, 2, 0)).copy()

    if isinstance(gtruthcoordinates, torch.FloatTensor):
        gtruthcoordinates = gtruthcoordinates.numpy()
    
    # Calculate euclidean
    headEuclid = euclidean(predcoordinates[0], np.array([gtruthcoordinates[0],gtruthcoordinates[1]]))
    tailEuclid = euclidean(predcoordinates[1], np.array([gtruthcoordinates[2],gtruthcoordinates[3]]))
    avgEuclid = 0.5*(headEuclid + tailEuclid)
    
    # Convert coordinates 
    height, width = image.shape[:2]
    predhead = (int (predcoordinates[0, 0] * width), int (predcoordinates[0, 1] * height))
    predtail = (int (predcoordinates[1, 0] * width), int (predcoordinates[1, 1] * height))

    gtruthhead = (int(gtruthcoordinates[0] * width), int(gtruthcoordinates[1] * height))
    gtruthtail = (int(gtruthcoordinates[2] * width), int(gtruthcoordinates[3] * height))

    cv2.arrowedLine(image, gtruthtail, gtruthhead, (1., 0., 0.), 3)
    cv2.arrowedLine (image, predtail, predhead, (0., 0., 1.), 3)
    plt.imshow(image)
    plt.axis('off')
    plt.title ('Head:{:.03f}\n Tail:{:.03f}\n Avg:{:.03f}'.format (headEuclid, tailEuclid, avgEuclid))
    plt.pause(0.001)


def show_arrow_batch(sample_batched):
    images_batch, coordinates_batch, cls_batch = \
        sample_batched['image'], sample_batched['coordinates'], sample_batched['cls']
    batch_size = len(images_batch)
    im_h, im_w = images_batch.size(2), images_batch.size(3)

    grid = utils.make_grid(images_batch)
    grid = grid.numpy().transpose((1, 2, 0))
    grid = grid.copy()

    for i in range(batch_size):
        hx, hy, tx, ty = coordinates_batch[i].numpy()
        head, tail = (int(tx * im_w + i * im_w), int(ty * im_h)), \
                     (int(hx * im_w + i * im_w), int(hy * im_h))
        cv2.arrowedLine(grid, tail, head, (1., 0., 0.), 3)

    plt.imshow(grid)
    # plt.title('  '.join(cls_batch))


def show_image_batch(images_batch):
    grid = utils.make_grid(images_batch)
    grid = grid.numpy().transpose((1, 2, 0))
    grid = grid.copy()

    plt.imshow(grid)

def showHeadTailDistribution(clsHeadTailEuclid, fullDataset=False):
    """
    Plot head & tail distribution for a given class

    example usage:
    clsHeadTailEuclid = {'Head Distribution': headEuclid, 'Tail Distribution': tailEuclid}
    showHeadTailDistribution(clsHeadTailEulcid, True)

    :param clsHeadTailEuclid: dictionary with key --> part name, value --> array of Euclidean distances
    :param fullDataset: boolean to decide number of bins
    :return:
    """
    if fullDataset:
        numBins = 500
    else:
        numBins = 50
    numParts = 2
    assert isinstance(clsHeadTailEuclid, dict)
    assert len(clsHeadTailEuclid) == numParts

    fig, axarr = plt.subplots(1, numParts, figsize=(15,4))
    for i, part in enumerate(clsHeadTailEuclid):
        clsMin, clsMean, clsMax, clsStd = clsHeadTailEuclid[part].min(), clsHeadTailEuclid[part].mean(), clsHeadTailEuclid[part].max(), np.std(clsHeadTailEuclid[part])
        axarr[i].hist(clsHeadTailEuclid[part], numBins, range=[0.0, 1.0])
        axarr[i].set_title(part)
        axarr[i].text (0.65, 0.85,
                           'Minimum: {:0.3f}\nMean: {:0.3f}\nMax: {:0.3f}\nStd: {:0.3f}'.format (clsMin, clsMean,
                                                                                                 clsMax, clsStd),
                           bbox=dict (facecolor='red', alpha=0.5),
                           horizontalalignment='center', verticalalignment='center', transform=axarr[i + 2].transAxes)

def showClassificationDistribution(data,title):
    fig, axarr = plt.subplots (1, 1, figsize=(15, 4))
    clsMin, clsMean, clsMax, clsStd = data.min(), data.mean(), data.max(), np.std(data)
    axarr.hist(clsHeadTailEuclid[part], numBins, range=[0.0, 1.0])
    axarr.set_title(part)
    axarr.text (0.65, 0.85,
                       'Minimum: {:0.3f}\nMean: {:0.3f}\nMax: {:0.3f}\nStd: {:0.3f}'.format (clsMin, clsMean,
                                                                                             clsMax, clsStd),
                       bbox=dict (facecolor='red', alpha=0.5),
                       horizontalalignment='center', verticalalignment='center', transform=axarr.transAxes)

def plotPoseVarKLDiv(filepath, metric, datasets, ylbl=None):

    """ TEMPORARY BEGIN """
    #TODO Incorporate calculating pose variability & kl divergence into dataset creation
    poseVar, kldiv, label = [], [], []
    PoseVarMetrics = pickle.load(open('tmp/PoseVarMetrics.p', 'rb'))
    for i in datasets:
        poseVar += [PoseVarMetrics[i][cls]['PoseVar'] for cls in PoseVarMetrics[i]]
        kldiv += [PoseVarMetrics[i][cls]['KLDiv'] for cls in PoseVarMetrics[i]]
        label += [dummyLbl for dummyLbl, cls in enumerate(PoseVarMetrics[0])]
    classes = PoseVarMetrics[0].keys()
    """ TEMPORARY END"""

    N = len(label)
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    bounds = np.linspace(0,N,N+1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    fig, axarr = plt.subplots(1,2, figsize=(15,4))
    scat = axarr[0].scatter(poseVar, metric, c=label, cmap=cmap, marker='x')
    axarr[0].set_xlabel('Pose Variability'); axarr[0].set_ylabel(ylbl)
    axarr[0].set_title('Pose Variability vs {} (Class by Class)'.format(ylbl))

    scat1 = axarr[1].scatter(kldiv, metric, c=label, marker='x', cmap=cmap)
    axarr[1].set_xlabel('KL Divergence Train & Test Set'); axarr[1].set_ylabel(ylbl)
    axarr[1].set_title('KL Divergence vs {} (Class by Class)'.format(ylbl))
    cb1 = plt.colorbar(scat1, spacing='proportional',ticks=bounds)
    cb1.set_label('Classes')
    cb1.set_ticklabels(classes)
    plt.savefig(filepath + '/posevar.png')

def showConfusionMatrix(filepath, cm, classes, checkpoint=None, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
    if normalize:
        cm = cm.astype ('float') / cm.sum (axis=1)[:, np.newaxis]

    plt.figure(figsize=(20,20))
    plt.imshow (cm, interpolation='nearest', cmap=cmap)
    plt.title (title)
    plt.colorbar ()
    tick_marks = np.arange (len (classes))
    plt.xticks (tick_marks, classes, rotation=45)
    plt.yticks (tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max () / 2.
    for i, j in itertools.product (range (cm.shape[0]), range (cm.shape[1])):
        plt.text (j, i, format (cm[i, j], fmt),
                  horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout ()
    plt.ylabel ('True label')
    plt.xlabel ('Predicted label')
    plt.savefig(filepath + '/cmatrix' + '_' + checkpoint[:-7] + 'png')

def plotPoseOrientation(filepath, predictions, gtruth, poseXY, classAccuracy, classIdx, specimenIDs):
    numCols = 8
    classes = classAccuracy.keys()
    order = np.argsort([classAccuracy[cls] for cls in classes])
    classes = [classes[i] for i in order]

    numRows = int(max(len(classes)/numCols+1,2))
    numPlots = int(np.ceil(numRows / 5.))
    for p in range(numPlots):
        fig, axarr = plt.subplots(min(numRows, 5), numCols, figsize=(20,20))
        for i in range(len(axarr)):
            for j in range(len(axarr[0])):
                axarr[i,j].set_axis_off()
        for iAx, cls in enumerate(classes[p*5*numCols:(p+1)*5*numCols]):
            print ' ', cls, classAccuracy[cls], len(classIdx[cls])
            axarr[iAx / numCols, iAx % numCols].set_title('{:0.3f} ({})'.format(classAccuracy[cls], cls.split()[0]))
            axarr[iAx / numCols, iAx % numCols].plot([0,0], [-1,1], 'r')
            axarr[iAx / numCols, iAx % numCols].plot([-1, 1], [0,0], 'r')
            axarr[iAx / numCols, iAx % numCols].plot(np.cos(np.arange(0, 2.1*np.pi, 0.1)), np.sin(np.arange(0, 2.1*np.pi, 0.1)), 'k')

            idx = classIdx[cls]
            poseNorm = specimen_normalization(poseXY[idx], [specimenIDs[i] for i in idx])

            misClassifiedPoseNorm = np.array([poseNorm[i] for i, j in enumerate(idx) if gtruth[j] != predictions[j]])
            correctClassifiedPoseNorm = np.array([poseNorm[i] for i, j in enumerate(idx) if gtruth[j] == predictions[j]])

            if not misClassifiedPoseNorm.shape[0] == 0:
                axarr[iAx / numCols, iAx % numCols].plot (misClassifiedPoseNorm[:, 0],
                                                              misClassifiedPoseNorm[:, 1], '.r')
            if not correctClassifiedPoseNorm.shape[0] == 0:
                axarr[iAx / numCols, iAx % numCols].plot(correctClassifiedPoseNorm[:, 0], correctClassifiedPoseNorm[:, 1], '.g')

            axarr[iAx / numCols, iAx % numCols].set_xlim(-1.1, 1.1)
            axarr[iAx / numCols, iAx % numCols].set_ylim(-1.1, 1.1)

    plt.savefig(filepath + '/orientationspace.png')
