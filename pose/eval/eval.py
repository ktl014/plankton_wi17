'''
eval.py

Created on Feb 21 2018 11:52
#@author: Kevin Le
'''

import numpy as np
import caffe
import glob
import timeit
import collections
from utils.data import group_specimen2class

"""
Initialize model to grab weights & prototxt
Use model to do following for each batch:
- Preprocess
- Estimate keypoints
- Compute keypoint coordinates

"""
ROOTDIR = '/data5/lekevin/plankton/poseprediction'

class PoseModelEvaluator(object):
    __names__ = ['alexnet', 'vgg19', 'resnet50']

    def __init__(self, gpuID, modelName):
        super(PoseModelEvaluator, self).__init__()

        # Check model to evaluate is allowed
        assert modelName in PoseModelEvaluator.__names__

        if modelName == 'alexnet':
            self.deployPrototxt = ROOTDIR + '/models/bvlc_alexnet/deploy.prototxt'
            self.trainedWeights = ROOTDIR + '/models/bvlc_alexnet/bvlc_alexnet.caffemodel'
        elif modelName == 'vgg19':
            self.deployPrototxt = ROOTDIR + '/models/vgg19/deploy.prototxt'
            self.trainedWeights = ROOTDIR + '/models/vgg19/VGG_ILSVRC_19_layers.caffemodel'
        elif modelName == 'resnet50':
            self.deployPrototxt = ROOTDIR + '/models/resnet-50/deploy.prototxt'
            self.trainedWeights = ROOTDIR + '/models/resnet-50/ResNet-50-model.caffemodel'

        self.gpuID = gpuID
        caffe.set_mode_gpu()
        caffe.set_device(self.gpuID)

    def loadPretrainedModel(self):
        self.net = caffe.Net(self.deployPrototxt, caffe.TEST, weights=self.trainedWeights)

    def estimateKeyPoints(self, batch):
        batchSize = len(batch)
        self.net.blobs['image'].data[:batchSize] = batch
        self.net.forward()
        # Keys ---> image

        #BEGIN: predMaps (checking each layer's predmap - TEMP) --> convert to only last background
        predKeys = ['conv5_5_CPM_L2'] + ['Mconv7_stage{}_L2'.format(i) for i in range(2,7)]
        predMaps = OrderedDict([(k, self.net.blobs[k].data[0]) for k in predKeys])
        #END

        poseCoords = np.stack([np.unravel_index(p.argmax(), p.shape) for p in predMaps])
        return poseCoords

if __name__ == '__main__':
    gpuDevice = 0
    modelToEval = 'vgg19'
    model = PoseModelEvaluator(gpuDevice, modelToEval)
    data = np.load('/data5/morgado/projects/plankton/turk/results/pose.npy').all()
    taxLvlDatasets = group_specimen2class(data['images'])

    model.loadPretrainedModel()
    '''
    poseCoords = OrderedDict() #---> imageFileName:poseCoord
    nSmpl = len(images)
    for i in range(0, nSmpl, 25):
        batch = [prepImg(img) for img in images[i:i+25]]
        poseCoords.append(model.estimateKeyPoints(batch))
        if i%1000 == 0:
            print('Samples computed:', i, '/', nSmpl)
            sys.stdout.flush()
    '''

    # Evaluate keypoints given the predicted coordinate
    # Set up function to set up the dataset
    for specimenSet in taxLvlDatasets:
         for cls in specimenSet:










