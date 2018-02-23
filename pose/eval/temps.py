import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.spatial.distance import euclidean

class Accuracy(object):

    def __init__(self):
        self.gtruth = np.array(([93, 16], [4,32]))

    def computeEuclideanDist(self, prediction):
        numParts = 2   # Part -> Head or Tail
        assert isinstance(prediction, (np.ndarray, np.generic))
        assert prediction.ndim == numParts

        distH = euclidean(prediction[0], self.gtruth[0])
        distT = euclidean(prediction[1], self.gtruth[1])
        euclidDist = np.array([distH, distT])
        return euclidDist

    def checkAccuracy(self):
        '''
        Compute Accuracy after all euclidean distance calculated
        :return:
        '''
        return 0

    def averageScore(self):
        return 0

    def retrieveGtruth_Temp(self):
        txtFile = '/data5/lekevin/plankton/poseprediction/code/20170124_001-timeseries.lst'
        imgFileNames = open(txtFile).read().splitlines()
        metaData = json.load('/data5/Plankton_wi18/rawcolor_db/meta/20170124_001-meta.json')
        imgFileNames = [fn for fn in metaData if 'annotation' in metaData]
        for fn in imgFileNames:
            pose = metaData[fn]['annotation']
            headOOF = [isinstance(p['head'], dict) for p in pose]
            tailOOF = [isinstance(p['tail'], dict) for p in pose]
            if not all(tailOOF) or not all(headOOF):
                continue
            head_x = int(np.median ([p['head']['x'] for p in pose]))
            head_y = int(np.median ([p['head']['y'] for p in pose]))
            tail_x = int(np.median ([p['tail']['x'] for p in pose]))
            tail_y = int(np.median ([p['tail']['y'] for p in pose]))


if __name__=='__main__':
    def computeEuclideanDist_tst():
        accu = Accuracy()
        prediction = np.random.randint(72, size=(2,2))
        distH, distT = accu.computeEuclideanDist(prediction)

    computeEuclideanDist_tst()