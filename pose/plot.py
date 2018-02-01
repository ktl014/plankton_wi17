#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 22:53:12 2018

@author: ktl014
"""
from utils import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
plt.rcParams['font.size'] = 9
plt.rcParams['figure.figsize'] = (12,19)

def main():
    batch0 = 'Batch_3084800_batch_results.csv'
    batch1 = 'batchResults/Batch_180498_batch_results.csv'
    batch2 = 'batchResults/Batch_180520_batch_results.csv'
    with open(batch2, 'r') as csv_file:
        data = read_csv(csv_file)

    spec = sorted(list(set(d['specimen'] for d in data)))
    clas = sorted(list(set(d['class'] for d in data)))
    specimen = dict(zip(spec,range(len(spec))))
    taxclass = dict(zip(clas,range(len(clas))))
    for i in specimen.iterkeys():
        ind = []
        for j, d in enumerate(data):
            if d['specimen'] == i:
                ind.append(j)
        specimen[i] = ind
        
    for i in taxclass.iterkeys():
        specind = [d['specimen'] for j,d in enumerate(data) if d['class']==i]
        taxclass[i] = list(set(specind))
    for k,v in taxclass.iteritems():
        for j,vv in enumerate(v):
            ind = [i for i,d in enumerate(data) if d['specimen'] == vv]
            taxclass[k][j] = {vv:ind}
#    ind = [i for i,d in enumerate(data) if d['specimen'] == specimenList[0]]

#    fig = plt.figure()
#    ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, 255), ylim=(0, 255))
    numRows = 6; numCol = 5; cntr = 1
    for k,v in taxclass.iteritems(): # Dict of classes
        fig = plt.figure();
        for j, vv in enumerate(v): # List of specimens
            for x,y in vv.iteritems(): # Dict specimens:indices
                ax = fig.add_subplot(numRows,numCol,j+1, autoscale_on=False, xlim=(0, 255), ylim=(0, 255))
                for t in y:
                    d = data[t]
                    if d['head'] == 'out of frame' or d['tail'] == 'out of frame':
                        continue
                    head, tail = (d['head']['x'], d['head']['y']), (d['tail']['x'], d['tail']['y'])
                    ax.annotate('', xy=(128 + head[0] - tail[0], 128 + head[1] - tail[1]),
                xytext=(128, 128), arrowprops=dict(arrowstyle="->", color='r', lw=1))
                plt.title(d['specimen'].split('_')[1]); plt.axis('off')
        print k + ' arrowmap generated, ' + str(cntr) + '/' + str(len(taxclass)) + ' remaining'
        fig.savefig('arrowmaps/' + d['class'] + '.png')
        cntr +=1
    plotAllSpecimens(data, specimen)
#        for i in range(len(taxclass[k])):
#            d = data[taxclass[k][i]]
#            if d['head'] == 'out of frame' or d['tail'] == 'out of frame':
#                continue
#            head, tail = (d['head']['x'], d['head']['y']), (d['tail']['x'], d['tail']['y'])
#            ax.annotate('', xy=(128 + head[0] - tail[0], 128 + head[1] - tail[1]),
#                xytext=(128, 128),
#                arrowprops=dict(arrowstyle="->", color='r', lw=1))
#        plt.title(d['specimen']); plt.axis('off')
#        print k + ' arrowmap generated, ' + str(j) + '/' + str(len(taxclass)) + ' remaining'
#        fig.savefig('arrowmaps/' + d['class'] + '.png')
    
#        if d['head'] == 'out of frame' and d['tail'] == 'out of frame':
#            framecount.append(i)
#        if d['focus'] == 'Slightly out of focus':
#            focuscount.append(i)
#    framecount = []
#    focuscount = []
#    plotData(data, framecount)
#    plt.savefig('frame.png')
#    plotData(data, focuscount)
#    plt.savefig('focus.png')
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
    fig.savefig('PoseVariance.png')

def plotData(data, ind):
    numRows = 9; numCols = 5
    plt.rcParams['font.size'] = 9
    plt.rcParams['figure.figsize'] = (12,19)
    plt.figure()
    for k in range(numRows*numCols):
        randInd = np.random.randint(len(ind))
        plt.subplot(numRows, numCols, k+1);
        img = cv2.imread(data[randInd]['img_file'])
        b,g,r = cv2.split(img)
        rgb_img = cv2.merge([r,g,b])
        plt.imshow(rgb_img)
        plt.title(data[randInd]['class']); plt.axis('off')

if __name__ == '__main__':
    main()