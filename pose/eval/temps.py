import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.spatial.distance import euclidean

class Accuracy(object):

    def __init__(self, headX, headY, tailX, tailY):
        assert headX.shape == headY.shape and tailX.shape == tailY.shape

        self.gtruthHead = np.column_stack((headX, headY))
        self.gtruthTail = np.column_stack((tailX, tailY))

        print self.gtruthHead.shape, self.gtruthTail.shape

    #TODO
    def euclideanDistance(self, prediction):
        numParts = 2   # Part -> Head or Tail
        assert isinstance(prediction, (np.ndarray, np.generic))
        assert prediction.ndim == numParts

        computeEuclideanDist_tst()

        return np.array(headEuclid, tailEuclid, avgEuclid)

    def computeEuclideanDist(self, prediction):

        distH = euclidean(prediction[0], self.gtruth[0])
        distT = euclidean(prediction[1], self.gtruth[1])
        euclidDist = np.array([distH, distT])
        return euclidDist

    def checkAccuracy(self):
        '''
        Compute Accuracy after all euclidean distance calculated
        :return:q
        '''
        return 0

    def averageScore(self):
        return 0

if __name__=='__main__':
    def computeEuclideanDist_tst():
        accu = Accuracy()
        prediction = np.random.randint(72, size=(2,2))
        distH, distT = accu.computeEuclideanDist(prediction)

    computeEuclideanDist_tst()