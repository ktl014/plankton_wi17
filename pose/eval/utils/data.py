import numpy as np
import pandas as pd
import csv
import cv2
import os
import glob
from constants import *


def url_to_filename(img_url):
    return img_url.split('/')[-1]


def url_to_class(img_url):
    return img_url.split('/')[5].split('_')[0]


def read_csv(csv_filename):
    with open(csv_filename, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data = []
        for row in csv_reader:
            annotation_data = eval(row['Answer.annotation_data'])
            for d in annotation_data:
                processed_data = preprocess(d)
                if processed_data is not None:
                    data.append(processed_data)
    return pd.DataFrame(data)


def preprocess(datum):
    if datum['head'] == 'out of frame' or datum['tail'] == 'out of frame':
        return

    new_datum = {}
    new_datum[IMG_NAME] = url_to_filename(datum['url'])
    new_datum[CLASS] = url_to_class(datum['url'])
    new_datum[COORDINATES] = [datum['head']['relative_x'],
                              datum['head']['relative_y'],
                              datum['tail']['relative_x'],
                              datum['tail']['relative_y']]

    return new_datum


def update_old(csv_filename, img_dir):
    df = pd.read_csv(csv_filename)
    for i, row in df.iterrows():
        ann_data = eval(row['Answer.annotation_data'])
        for d in ann_data:
            img_file = os.path.join(img_dir,
                                    url_to_filename(d['url']))
            img = cv2.imread(img_file)
            height, width = img.shape[:2]
            turk_width = 200
            turk_height = int(round(height * turk_width / width))
            for p in ['head', 'tail']:
                if d[p] == 'out of frame':
                    continue
                d[p]['click_x'], d[p]['click_y'] = d[p]['x'], d[p]['y']
                d[p]['relative_x'] = d[p]['x'] / turk_width
                d[p]['relative_y'] = d[p]['y'] / turk_height
                d[p]['width'], d[p]['height'] = turk_width, turk_height
                del d[p]['x'], d[p]['y']
            del d['width'], d['height']
        df.loc[i, 'Answer.annotation_data'] = str(ann_data)
    return df


def data_filter(data):
    grouped = data.groupby(IMG_NAME)
    coordinates_group = grouped[COORDINATES].apply(list).apply(lambda x: np.median(x, axis=0))
    class_group = grouped[CLASS].apply(list).apply(lambda x: x[0])
    data = pd.concat([coordinates_group, class_group], axis=1)
    return data.reset_index()

def group_specimen2class(imgList):
    specimenIDs = [img.split('/')[0] for img in imgList]
    planktonLabels = plankton_labels()

    specimenSet = {spc: [spc] for spc in set(specimenIDs)}     # Specimen Lvl
    genus = [planktonLabels[spc][2] for spc in specimenIDs]
    genusSet = {cls: [spc for spc in specimenIDs if planktonLabels[spc][2] == cls] for cls in set(genus)}   # Genus Lvl
    family = [planktonLabels[spc][1] for spc in specimenIDs]
    familySet = {cls: [spc for spc in specimenIDs if planktonLabels[spc][1]==cls] for cls in set(family)}   # Family Lvl
    order = [planktonLabels[spc][0] for spc in specimenIDs]
    orderSet = {cls: [spc for spc in specimenIDs if planktonLabels[spc][0]==cls] for cls in set(order)}   # Order Lvl
    dataSet = {'Dataset': specimenIDs}
    return [specimenSet, genusSet, familySet, orderSet, dataSet]


def plankton_labels():
    # Load all specimens
    specimen_list = ['_'.join (fn.split ('/')[-2:]) for fn in
                     glob.glob ('/data4/plankton_wi17/plankton/images_orig/*/*')]

    # Load all labels
    specimen_labels = [l.split ('\t') for l in open ('/data5/Plankton_wi18/specimen_taxonomy.txt').read ().splitlines ()[1:]]
    specimen_labels = {l[0]: l[1:] for l in specimen_labels if not l[0].startswith ('google')}
    for spc in specimen_list:
        if spc not in specimen_labels:
            specimen_labels[spc] = ['Unknown'] * 3

    return specimen_labels