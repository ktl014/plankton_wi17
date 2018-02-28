import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean

class Accuracy(object):

    def __init__(self, headX, headY, tailX, tailY):
        assert headX.shape == headY.shape and tailX.shape == tailY.shape

        self.gtruthHead = np.column_stack((headX, headY))
        self.gtruthTail = np.column_stack((tailX, tailY))
        self.poseX = headX - tailX
        self.poseY = headY - tailY

    def euclideanDistance(self, prediction):
        numParts = 2   # Part -> Head or Tail
        if not isinstance(prediction, (np.ndarray, np.generic)):
            prediction = np.asarray(prediction)
        # assert prediction.ndim == numParts

        headEuclid = []; tailEuclid =[]
        for i in range(len(prediction)):
            headEuclid.append(euclidean(prediction[i][0], self.gtruthHead[i]))
            tailEuclid.append(euclidean(prediction[i][1], self.gtruthTail[i]))
        histData = {'Head Distribution':np.asarray(headEuclid), 'Tail Distribution':np.asarray(tailEuclid)}
        avgHeadEuclid = headEuclid.mean()
        avgTailEuclid = tailEuclid.mean()
        avgEuclid = np.asarray(avgHeadEuclid, avgTailEuclid).mean()
        return [avgHeadEuclid, avgTailEuclid, avgEuclid, histData]

    def plotEuclidDistribution(self, clsMetrics, clsMetIdx, data):
        img_dir = '/data5/Plankton_wi18/rawcolor_db/images'
        numCols = 3

        # Order classes
        classes = clsMetrics.keys()
        order = np.argsort([clsMetrics[cls] for cls in classes])
        classes = [classes[i] for i in order]
        avgEuclid = 2
        histData = 4

        #plot Distributions
        for cls in classes:
            fig, axarr = plt.subplot(1, numCols, figsize=(15,4))
            img = plt.imread(os.path.join(img_dir,data[clsMetIdx[cls][0]]))
            axarr[0].imshow(img)
            axarr[0].set_axis_off()
            axarr[0].set_title('{:.03f} ({})'.format(clsMetrics[avgEuclid], cls.split()[0]))
            clsMin, clsMean, clsMax, clsStd = euclidStats(self, clsMetrics[cls][histData])
            for i,part in enumerate(histData):
                axarr[i+1].hist(histData[[part]])
                axarr[i+1].set_title(part)
                axarr[i+1].text(1,5000, 'Minimum: {}\nMean: {}\nMax: {}\nStd: {}'.format(clsMin, clsMean, clsMax, clsStd), bbox=dict(facecolor='red', alpha=0.5),
                                horizontalalignment='center', verticalalignment='center')

    def euclidStats(self, histData):
        return histData.min(), histData.mean(), histData.max(), np.std(histData)

if __name__=='__main__':
    def computeEuclideanDist_tst():
        accu = Accuracy()
        prediction = np.random.randint(72, size=(2,2))
        distH, distT = accu.computeEuclideanDist(prediction)

