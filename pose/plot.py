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
import glob
plt.rcParams['font.size'] = 9
plt.rcParams['figure.figsize'] = (12,19)

def main():
#%%
    batch0 = 'Batch_3084800_batch_results.csv'
    batch1 = 'batchResults/Batch_180498_batch_results.csv'
    batch2 = 'batchResults/Batch_180520_batch_results.csv'
    with open(batch0, 'r') as csv_file:
        old_data = read_csv(csv_file)
#%%
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
    plotData(data, old_ddata['frame']); plt.savefig('frame.png')
#    plotData(data, old_ddata['z-dir']['parallel']); plt.savefig('z-dir/parallel.png')
#    plotData(data, old_ddata['z-dir']['towardCam']); plt.savefig('z-dir/towardCam.png')
#    plotData(data, old_ddata['z-dir']['awayCam']); plt.savefig('z-dir/awayCam.png')
#    plotData(data, old_ddata['z-dir']['notSure']); plt.savefig('z-dir/notSure.png')
#    
    
    plotData(data, ddata['focus']['infocus']); plt.savefig('infocus_filt.png')
    plotData(data, ddata['focus']['slightfocus']); plt.savefig('slightfocus_filt.png')
    plotData(data, ddata['focus']['outfocus']); plt.savefig('outfocus_filt.png')
    plotData(data, ddata['z-dir']['parallel']); plt.savefig('z-dir/parallel_filt.png')
    plotData(data, ddata['z-dir']['towardCam']); plt.savefig('z-dir/towardCam_filt.png')
    plotData(data, ddata['z-dir']['awayCam']); plt.savefig('z-dir/awayCam_filt.png')
    plotData(data, ddata['z-dir']['notSure']); plt.savefig('z-dir/notSure_filt.png')

        
    taxclass, specimen = setImgBySpecClass(data)
    #plotSpecimensByClass(data, taxclass)
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

def plotSpecimensByClass(data, taxclass):
    numRows1 = 5; numCols1 = 3
    numRows = 6; numCol = 5; cntr = 1
    fig1 = plt.figure(); specfig1 = plt.figure()
    for k,v in taxclass.iteritems(): # Dict of classes
        fig = plt.figure(); specfig = plt.figure()
        imgcnt = 0
        for j, vv in enumerate(v): # List of specimens
            for x,y in vv.iteritems(): # Dict specimens:indices
                ax = fig.add_subplot(numRows,numCol,j+1, autoscale_on=False, xlim=(0, 255), ylim=(0, 255))
                ax1 = fig1.add_subplot(numRows1, numCols1, cntr, autoscale_on=False, xlim=(0, 255), ylim=(0, 255))
                for t in y:
                    d = data[t]
                    if d['head'] == 'out of frame' or d['tail'] == 'out of frame':
                        continue
                    head, tail = (d['head']['x'], d['head']['y']), (d['tail']['x'], d['tail']['y'])
                    conv_head, conv_tail = (d['trueimg']['x']
                    ax.annotate('', xy=(128 + head[0] - tail[0], 128 + head[1] - tail[1]),xytext=(128, 128), arrowprops=dict(arrowstyle="->", color='r', lw=1))
                    ax1.annotate('', xy=(128 + head[0] - tail[0], 128 + head[1] - tail[1]),xytext=(128, 128), arrowprops=dict(arrowstyle="->", color='r', lw=1))
                    imgcnt +=1
                ax.set_title(d['specimen'].split('_')[1]); plt.axis('off')
                ax1.set_title(d['class']); plt.axis('off')
        print k + ' arrowmap generated, ' + str(imgcnt) + ' total imgs, ' + str(cntr) + '/' + str(len(taxclass)) + ' remaining'
        fig.savefig('arrowmaps/' + d['class'] + '.png')
        cntr +=1
    fig1.savefig('arrowmaps/allclasses.png')

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
#%%
if __name__ == '__main__':
    main()