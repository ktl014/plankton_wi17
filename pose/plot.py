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

def main():
    with open('Batch_3084800_batch_results.csv', 'r') as csv_file:
        data = read_csv(csv_file)
    framecount = []
    focuscount = []
    spec = sorted(list(set(d['specimen'] for d in data)))
    specimen = dict(zip(spec,range(len(spec))))
    for i in specimen.iterkeys():
        ind = []
        for j, d in enumerate(data):
            if d['specimen'] == i:
                ind.append(j)
        specimen[i] = ind
#    ind = [i for i,d in enumerate(data) if d['specimen'] == specimenList[0]]
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, 255), ylim=(0, 255))
    for i in range(len(ind)):
        d = data[ind[i]]
        if d['head'] == 'out of frame' or d['tail'] == 'out of frame':
            continue
        head, tail = (d['head']['x'], d['head']['y']), (d['tail']['x'], d['tail']['y'])
        ax.annotate('', xy=(128 + head[0] - tail[0], 128 + head[1] - tail[1]),
            xytext=(128, 128),
            arrowprops=dict(arrowstyle="->", color='r', lw=1))
    fig.savefig(specimenList[0]+'.png')
    
    
#        if d['head'] == 'out of frame' and d['tail'] == 'out of frame':
#            framecount.append(i)
#        if d['focus'] == 'Slightly out of focus':
#            focuscount.append(i)
#    plotData(data, framecount)
#    plt.savefig('frame.png')
#    plotData(data, focuscount)
#    plt.savefig('focus.png')

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