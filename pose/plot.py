#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 22:53:12 2018

@author: ktl014
"""
from utils import *
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import cv2
import seaborn as sns
plt.rcParams['font.size'] = 9
#plt.rcParams['figure.figsize'] = (12,19)

def main():
#%%
    batch0 = 'batchResults/Batch_3084800_batch_results.csv'
    batch1 = 'batchResults/Batch_180498_batch_results.csv'
    batch2 = 'batchResults/Batch_180520_batch_results.csv'
    with open(batch0, 'r') as csv_file:
        old_data = read_csv(csv_file)

    print 'Total annotations: {} by 410 workers'.format(len(old_data))
    image_data = defaultdict(list)
    for d in old_data:
        img_name = d['url'].split('/')[-1]
        image_data[img_name].append(d)
    results = []
    for image in image_data:
        r = filter_results(image_data[image])
        if r is not None:
            results.append(r)
    data = results
    print 'Total annotations: {}'.format(len(data))
    
#%%    
    old_ddata = plotConfidenceScore(old_data)
    print('Unfiltered data')
    print 'Confident Workers: {} / {}'.format(len(old_ddata['confidence']), len(old_data))
    #print '{} "out of focus" images / {}'.format(len(old_ddata['focus']), int(len(old_data)))
    print '{} "out of frame" images / {}'.format(len(old_ddata['frame']), int(len(old_ddata['confidence'])))
    for k,v in old_ddata['focus'].iteritems():
        print '{} "{}" images / {} ~ {}'.format(len(v), k, len(old_ddata['confidence']), len(v)/float(len(old_ddata['confidence']))*100)
    print '\n'
    for k,v in old_ddata['z-dir'].iteritems():
        print k + ' total: ' + str(len(v)) + ' ~ {}'.format(len(v)/float(len(old_ddata['confidence']))*100)
    
    ddata = plotConfidenceScore(data)
    print('\nFiltered data')
    print 'Confident Workers: {} / {}'.format(len(ddata['confidence']), len(data))
    #print '{} "out of focus" images / {}'.format(len(ddata['focus']), int(len(data)))
    print '{} "out of frame" images / {}'.format(len(ddata['frame']), int(len(ddata['confidence'])))
    for k,v in ddata['focus'].iteritems():
        print '{} "{}" images / {} ~ {}'.format(len(v), k, len(ddata['confidence']), len(v)/float(len(ddata['confidence']))*100)
    print '\n'
    for k,v in ddata['z-dir'].iteritems():
        print k + ' total: ' + str(len(v)) + ' ~ {}'.format(len(v)/float(len(ddata['confidence']))*100)
        
#%%        
#    plotData(data, old_ddata['frame']); plt.savefig('frame.png')
##    plotData(data, old_ddata['z-dir']['parallel']); plt.savefig('z-dir/parallel.png')
##    plotData(data, old_ddata['z-dir']['towardCam']); plt.savefig('z-dir/towardCam.png')
##    plotData(data, old_ddata['z-dir']['awayCam']); plt.savefig('z-dir/awayCam.png')
##    plotData(data, old_ddata['z-dir']['notSure']); plt.savefig('z-dir/notSure.png')
##    
#    
#    plotData(data, ddata['focus']['infocus']); plt.savefig('infocus_filt.png')
#    plotData(data, ddata['focus']['slightfocus']); plt.savefig('slightfocus_filt.png')
#    plotData(data, ddata['focus']['outfocus']); plt.savefig('outfocus_filt.png')
#    plotData(data, ddata['z-dir']['parallel']); plt.savefig('z-dir/parallel_filt.png')
#    plotData(data, ddata['z-dir']['towardCam']); plt.savefig('z-dir/towardCam_filt.png')
#    plotData(data, ddata['z-dir']['awayCam']); plt.savefig('z-dir/awayCam_filt.png')
#    plotData(data, ddata['z-dir']['notSure']); plt.savefig('z-dir/notSure_filt.png')

        
    taxclass, specimen = setImgBySpecClass(data)
    #plotHisto(data, taxclass)
    plotSpecimensByClass(data, taxclass)
    #plotAllSpecimens(data, specimen)
    
    

#    framecount = []
#    focuscount = []
#    plotData(data, framecount)
#    plt.savefig('frame.png')
#    plotData(data, focuscount)
#    plt.savefig('focus.png')
#%%    
def plotConfidenceScore(data):
    framecount = []
    infocuscount = []; slifocuscount = []; outfocuscount = [];
    conficount = []
    parallelcount = []
    towardCamcount = []
    awayCamcount = []
    notsureCamcount = []
    for i,d in enumerate(data):
        if d['confidence'] == 'Very Confident' or d['confidence'] == 'Confident':
            conficount.append(i)
            if d['head'] == 'out of frame' and d['tail'] == 'out of frame':
                framecount.append(i)
            if d['focus'] == 'In focus':
                infocuscount.append(i)
            elif d['focus'] == 'Slightly out of focus':
                slifocuscount.append(i)
            elif d['focus'] == 'Out of focus':
                outfocuscount.append(i)
                
            if d['z-dir'] == 'Parallel':
                parallelcount.append(i)
            elif d['z-dir'] == 'Leaning toward camera':
                towardCamcount.append(i)
            elif d['z-dir'] == 'Leaning away from camera':
                awayCamcount.append(i)
            elif d['z-dir'] == 'Not Sure':
                notsureCamcount.append(i)
    ddata = {'confidence':conficount, 'frame':framecount, 
             'z-dir':{'parallel':parallelcount, 'towardCam':towardCamcount, 'awayCam': awayCamcount, 'notSure': notsureCamcount},
             'focus':{'infocus':infocuscount, 'slightfocus':slifocuscount, 'outfocus':outfocuscount}}
    return ddata
#%%    
def setImgBySpecClass(data):
    spec = sorted(list(set(d['specimen'] for d in data)))
    clas = sorted(list(set(d['class'] for d in data)))
    specimen = dict(zip(spec,range(len(spec))))
    taxclass = dict(zip(clas,range(len(clas))))
    
    # Get indices of each specimen
    for i in specimen.iterkeys():
        ind = []
        for j, d in enumerate(data):
            if d['specimen'] == i:
                ind.append(j)
        specimen[i] = ind
        
    # Organize specimens by class
    for i in taxclass.iterkeys():
        specind = [d['specimen'] for j,d in enumerate(data) if d['class']==i]
        taxclass[i] = list(set(specind))
    
    # Associate image files with each specimen of each class
    for k,v in taxclass.iteritems():    
        for j,vv in enumerate(v):
            ind = [i for i,d in enumerate(data) if d['specimen'] == vv]
            taxclass[k][j] = {vv:ind}
    return taxclass, specimen
#%%
def convertTrueXY(turkHT, trueDim, turkDim):
    assert isinstance (turkHT, dict) or isinstance(trueDim, dict) or isinstance(turkDim, dict)
    scale = trueDim['x']/200.0
    trueX = int(turkHT['x'] * scale) #trueDim['x'] * turkDim['x']
    trueY = int(turkHT['y'] * scale) #trueDim['y'] * turkDim['y']
    return (trueX, trueY)
#%%
# sns.distplot(theta, bins=20, kde=False, rug=True); plt.title('Toranidae_Specimen00 Theta'); plt.xlabel('Theta value'); plt.ylabel('Frequency');
# sns.distplot(radii, kde=False, rug=True); plt.title('Toranidae_Specimen00 Radius'); plt.xlabel('Radius value (True Pixel Coordinates)'); plt.ylabel('Frequency');    
def plotHisto(data, taxclass):
    fig = plt.figure(); 
    numRows = 5; numCols = 3; cntr = 1
    for k,v in taxclass.iteritems(): # Dict of classes vs specimens
        radii = []; theta = []
        ax = fig.add_subplot(numRows, numCols, cntr, autoscale_on=True) #, xlim=(-3, 3) , ylim=(0, 255))
        fig1 = plt.figure()
        for i,vv in enumerate(v): # List of specimens
            totSubplot = len(v)
            totCols = 4
            totRows = totSubplot // totCols
            totRows += totSubplot % totCols
            ax1 = fig1.add_subplot(totRows, totCols, i+1, autoscale_on=True)
            for x,y in vv.iteritems():  # Dict specimens:indices
                for t in y:
                    radii.append(data[t]['radius'])
                    theta.append(data[t]['theta'])
                radii1 = np.asarray([data[t]['radius'] for t in y])
                theta1 = np.asarray([data[t]['theta'] for t in y])
                #sns.distplot(radii1,kde=False, rug=True); plt.title(data[t]['specimen']); plt.xlabel('Radius Value'); plt.ylabel('Frequency'); plt.tight_layout(); sns.plt.xlim(radii1.min(), radii1.max());
                #plt.savefig('plots/radius/'+data[t]['class']+'.png')
                sns.distplot(theta1,kde=False, rug=True); plt.title(data[t]['class'] +'({})'.format(str(len(v)))); plt.xlabel('Theta Value'); plt.ylabel('Frequency'); plt.tight_layout(); sns.plt.xlim(theta1.min(), theta1.max());
                #plt.savefig('plots/angle/'+data[t]['class']+'.png')
        radii = np.asarray(radii)
        theta = np.asarray(theta)
        #sns.distplot(theta,kde=False, rug=True); plt.title(data[t]['class'] +'({})'.format(str(len(v)))); plt.xlabel('Theta Value'); plt.ylabel('Frequency'); plt.tight_layout();
        #sns.plt.xlim(theta.min(), theta.max());
        #sns.distplot(radii,kde=False, rug=True); plt.title(data[t]['class'] +'({})'.format(str(len(v)))); plt.xlabel('Radius Value'); plt.ylabel('Frequency'); plt.tight_layout();
        #sns.plt.xlim(radii.min(), radii.max());
        cntr += 1
#%%
def plotPolarHisto(data, taxclass):
    fig = plt.figure(); 
    numRows = 5; numCols = 3; cntr = 1
    for k,v in taxclass.iteritems(): # Dict of classes vs specimens
        radii = []; theta = []
        ax = fig.add_subplot(numRows, numCols, cntr, autoscale_on=True, projection='polar') #, xlim=(-3, 3) , ylim=(0, 255))
        for vv in v: # List of specimens
            for x,y in vv.iteritems():  # Dict specimens:indices
                for t in y:
                    radii.append(data[t]['radius'])
                    theta.append(data[t]['theta'])
                #radii = np.asarray([data[t]['radius'] for t in y])
                #theta.append([data[t]['theta'] for t in y])
        radii = np.asarray(radii)
        theta = np.asarray(theta)
        width = np.pi / 4 * np.random.rand(len(radii))
        bars = ax.bar(theta, radii, width=width, bottom=0.0)
        for r, bar in zip(radii, bars):
            bar.set_facecolor(plt.cm.viridis(r / 10.))
            bar.set_alpha(0.5)
        plt.title(data[t]['class'] +'({})'.format(str(len(v)))); plt.tight_layout();
        cntr += 1
#%%
def plotSpecimensByClass(data, taxclass):
    numRows1 = 5; numCols1 = 3; cntr = 1
    fig1 = plt.figure(); 
    for k,v in taxclass.iteritems(): # Dict of classes
        fig = plt.figure();
        imgcnt = 0
        for j, vv in enumerate(v): # List of specimens
            numCols = 3
            numRows = len(v) // numCols
            numRows += len(v) % numCols
            for x,y in vv.iteritems(): # Dict specimens:indices
                ax = fig.add_subplot(numRows,numCols,j+1, autoscale_on=False, xlim=(0, 255), ylim=(0, 255))
                ax1 = fig1.add_subplot(numRows1, numCols1, cntr, autoscale_on=False, xlim=(0, 255), ylim=(0, 255))
                for t in y:
                    d = data[t]
                    if d['head'] == 'out of frame' or d['tail'] == 'out of frame':
                        continue
#                    turkDim = {'x':d['width'], 'y':d['height']}
#                    head = convertTrueXY(d['head'], d['trueimg'], turkDim)
#                    tail = convertTrueXY(d['tail'], d['trueimg'], turkDim)
                    head, tail = d['true_head'], d['true_tail']
#                    trueX = d['head']['x'] * d['trueimg']['x'] / d['width']
#                    head, tail = (d['head']['x'], d['head']['y']), (d['tail']['x'], d['tail']['y'])
#                    conv_head, conv_tail = (d['trueimg']['x']
                    ax.annotate('', xy=(128 + head[0] - tail[0], 128 + head[1] - tail[1]),xytext=(128, 128), arrowprops=dict(arrowstyle="->", color='r', lw=1))
                    ax1.annotate('', xy=(128 + head[0] - tail[0], 128 + head[1] - tail[1]),xytext=(128, 128), arrowprops=dict(arrowstyle="->", color='r', lw=1))
                    imgcnt +=1
                ax.set_title(d['specimen'].split('_')[1]); plt.tight_layout()
                ax1.set_title(d['class'] + '({})'.format(str(len(v)))); plt.tight_layout()
        print k + ' arrowmap generated, ' + str(imgcnt) + ' total imgs, ' + str(cntr) + '/' + str(len(taxclass)) + ' remaining'
        fig.savefig('plots/arrowmaps/' + d['class'] + '.png')
        cntr +=1
    fig1.savefig('plots/arrowmaps/allclasses.png')
#%%
def plotAllSpecimens(data, specimen):
    fig = plt.figure()
    numRows = 3; numCols = 2; itr=0
    for k,v in specimen.iteritems():
        ax = fig.add_subplot(numRows,numCols,itr+1, autoscale_on=False, xlim=(0, 255), ylim=(0, 255))
        for i in v:
            d = data[i]
            if d['head'] == 'out of frame' or d['tail'] == 'out of frame':
                continue
            head, tail = (d['head']['click_x'], d['head']['click_y']), (d['tail']['click_x'], d['tail']['click_y'])
            ax.annotate('', xy=(128 + head[0] - tail[0], 128 + head[1] - tail[1]),
                xytext=(128, 128),
                arrowprops=dict(arrowstyle="->", color='r', lw=1))
        plt.title(d['specimen']); plt.axis('off')
        print k + ' arrowmap generated, ' + str(itr+1) + '/' + str(len(specimen)) + ' remaining'
        itr += 1
    #fig.savefig('smplallspecimens.png')
#%%
#plot N specimen by class
# plotData(data, taxclass['Toranidae'][0]['Toranidae_specimen00']) 
def plotData(data, ind):
    plotArrow = True
    numRows = 9; numCols = 5
    plt.rcParams['font.size'] = 9
    plt.rcParams['figure.figsize'] = (12,19)
    plt.figure()
    #imgFile = []
    for k in range(numRows*numCols):
        #randInd = np.random.randint(len(ind))
        d = data[ind[k]]
        if d['head'] == 'out of frame' or d['tail'] == 'out of frame':
            continue
        plt.subplot(numRows, numCols, k+1);
        img = cv2.imread(d['img_file'])
        #imgFile.append(d['img_file'])
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        if plotArrow:
#            height, width = img.shape[0], img.shape[1]
#            scale = width / 200.0
#            head = (int(d['head']['x'] * scale), int(d['head']['y'] * scale))
#            tail = (int(d['tail']['x'] * scale), int(d['tail']['y'] * scale))
            #print d['true_head'], d['true_tail']
            cv2.arrowedLine(img, d['true_tail'], d['true_head'], (0, 0, 255), 3)
        plt.imshow(img)
        plt.title(d['class']); plt.axis('off')
    #plt.savefig('toranidae.png')
    #return imgFile
#%%
if __name__ == '__main__':
    main()