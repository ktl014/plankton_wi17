import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
from astropy.modeling.models import Gaussian2D
import csv
import cv2
import os
from PIL import Image
from constants import *
import glob
import scipy.stats as sps
from scipy.spatial.distance import euclidean



def url_to_filename(img_url):
    return img_url.split('/')[-2:]


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


def coordinates_to_gaussian_map(coordinates, output_size, amplitude, sigma):
    width, height = np.arange(output_size[0]), np.arange(output_size[1])
    x, y = np.meshgrid(width, height)
    gaussian2d = Gaussian2D(amplitude, int(np.round(coordinates[0] * output_size[0])), int(round(coordinates[1] * output_size[1])), sigma, sigma)

    return gaussian2d(x, y)


def get_belief_map(coordinates, output_size, amplitude, sigma):
    head_map = coordinates_to_gaussian_map(coordinates[:2], output_size, amplitude, sigma)
    tail_map = coordinates_to_gaussian_map(coordinates[2:], output_size, amplitude, sigma)
    bg_map = np.maximum(1 - head_map - tail_map, 0)

    return np.asarray([head_map, tail_map, bg_map], dtype=np.float32)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def eval_euc_dists(pred_maps, targets):
    pred = [np.argwhere(mp == np.max(mp))[0, ::-1]
            for pred_map in pred_maps
            for mp in pred_map[:2]]
    pred = np.array(pred).reshape(targets.shape)

    w, h = pred_maps.shape[-2:]
    targets = targets * [w, h, w, h]

    dists = np.square(pred - targets)
    head_dist = np.mean(np.sum(dists[:, :2], axis=1))
    tail_dist = np.mean(np.sum(dists[:, 2:], axis=1))
    avg_dist = 0.5 * (head_dist + tail_dist)

    return {'head': head_dist, 'tail': tail_dist, 'average': avg_dist}


def get_output_size(model, input_size):
    inputs = torch.randn(1, 3, input_size, input_size)
    y = model(Variable(inputs))
    try:
        if isinstance(y, torch.autograd.variable.Variable):
            return y.size(-1)
        elif isinstance (y, tuple):
            return y[1].size(-1)
        else:
            raise TypeError
    except:
        print('ERROR @ utils.get_output_size(): Invalid type returned from output of model')
        assert False

def group_specimen2class(imgList, LEVEL):
    specimen_ids = [img.split('/')[0] for img in imgList]
    planktonLabels = plankton_labels()

    # specimenSet = {spc: [spc] for spc in set(specimenIDs)}     # Specimen Lvl
    # genus = [planktonLabels[spc][2] for spc in specimenIDs]
    # genusSet = {cls: [spc for spc in specimenIDs if planktonLabels[spc][2] == cls] for cls in set(genus)}   # Genus Lvl
    # family = [planktonLabels[spc][1] for spc in specimenIDs]
    # familySet = {cls: [spc for spc in specimenIDs if planktonLabels[spc][1]==cls] for cls in set(family)}   # Family Lvl
    # order = [planktonLabels[spc][0] for spc in specimenIDs]
    # orderSet = {cls: [spc for spc in specimenIDs if planktonLabels[spc][0]==cls] for cls in set(order)}   # Order Lvl
    # dataSet = {'Dataset': specimenIDs}
    if LEVEL.upper() == 'SPECIMEN':
        specimen_sets = {spc: [spc] for spc in set (specimen_ids)}
    elif LEVEL.upper() == 'GENUS':
        genus = [planktonLabels[spc][2] for spc in specimen_ids]
        specimen_sets = {cls: [spc for spc in specimen_ids if planktonLabels[spc][2] == cls] for cls in
                         set (genus)}
    elif LEVEL.upper() == 'FAMILY':
        family = [planktonLabels[spc][1] for spc in specimen_ids]
        specimen_sets = {cls: [spc for spc in specimen_ids if planktonLabels[spc][1] == cls] for cls in
                         set (family)}
    elif LEVEL.upper() == 'ORDER':
        family = [planktonLabels[spc][0] for spc in specimen_ids]
        specimen_sets = {cls: [spc for spc in specimen_ids if planktonLabels[spc][0] == cls] for cls in
                         set (family)}
    else:
        specimen_sets = {'Dataset': specimen_ids}
    # return {'Species':specimenSet, 'Genus':genusSet, 'Family':familySet, 'Order':orderSet,'Dataset':dataSet}, specimenIDs
    return specimen_sets, specimen_ids


def plankton_labels():
    # Load all specimens
    specimen_list = ['_'.join (fn.split ('/')[-2:]) for fn in
                     glob.glob ('/data4/plankton_wi17/plankton/images_orig/*/*')]

    # Load all labels
    specimen_labels = [l.split (',') for l in open ('/data5/Plankton_wi18/rawcolor_db2/classes/specimen_taxonomy.txt').read ().splitlines ()[1:]]
    specimen_labels = {l[0]: l[1:] for l in specimen_labels if not l[0].startswith ('google')}
    for spc in specimen_list:
        if spc not in specimen_labels:
            specimen_labels[spc] = ['Unknown'] * 3

    return specimen_labels

def xyz2sph(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    lng = np.arctan2(-x, z)
    lat = np.arctan2(-y, z)
    return r, lng, lat


def sph2xyz(r, lng, lat):
    x = - r * np.sin(lng) * np.cos(lat)
    y = - r * np.sin(lat)
    z = r * np.cos(lng) * np.cos(lat)
    return x, y, z


def equal_area_spherical_grid(long_bins=10, latt_bins=10):
    long_grid = np.linspace(-np.pi/2, np.pi/2, long_bins+1)
    y_grid = np.linspace(-1, 1, latt_bins+1)
    latt_grid = np.arcsin(y_grid)
    return long_grid, latt_grid


def estimate_sph_pose(pose_x, pose_y, specimen_ids):
    # Estimate z coordinate of pose vector based on estimated specimen size
    pose_z = np.zeros_like(pose_x)
    for spc in set(specimen_ids):
        idx = [i for i, spc_tmp in enumerate(specimen_ids) if spc_tmp == spc]
        dist = np.sqrt(pose_x[idx] ** 2 + pose_y[idx] ** 2)
        # spc_size = np.max(dist)
        spc_size = np.percentile(dist, 95)
        pose_z[idx] = np.sqrt(np.maximum(spc_size ** 2 - dist ** 2, 0))

    # Compute longitude and latitude angle of pose vector
    pose_rad, pose_lng, pose_lat = xyz2sph(pose_x, pose_y, pose_z)

    return pose_rad, pose_lng, pose_lat


def pose_variability(head_x, head_y, tail_x, tail_y, specimen_ids):
    # Compute pose vector
    pose_x = head_x - tail_x
    pose_y = head_y - tail_y

    # Compute longitude and latitude angle of pose vector
    _, pose_lng, pose_lat = estimate_sph_pose(pose_x, pose_y, specimen_ids)

    # Histogram grid (cylindrical equal-area projection)
    long_grid, latt_grid = equal_area_spherical_grid(long_bins=10, latt_bins=10)

    # Compute histogram
    h = np.zeros((long_grid.size-1, long_grid.size - 1))
    for lg in range(long_grid.size-1):
        for lt in range(long_grid.size - 1):
            long_t = (long_grid[lg] <= pose_lng) * (pose_lng < long_grid[lg + 1])
            latt_t = (latt_grid[lt] <= pose_lat) * (pose_lat < latt_grid[lt + 1])
            h[lg, lt] = (long_t * latt_t).sum()
    # print h

    # Compute probability distribution
    prob = h / h.sum()
    # print prob

    # Compute entropy
    entropy = sps.entropy(prob.reshape(-1)) / np.log(10)
    return entropy


def specimen_normalization(pose, specimen_ids):
    spc_size = np.zeros((pose.shape[0],))
    for spc in set(specimen_ids):
        idx = [i for i, spc_tmp in enumerate(specimen_ids) if spc_tmp == spc]
        dist = np.sqrt((pose[idx] ** 2).sum(axis=1))
        spc_size[idx] = np.percentile(dist, 95)
    pose_norm = pose / spc_size[:, np.newaxis]
    return pose_norm


def pose_histogram2(pose, specimen_ids):
    # Normalize pose by specimen size
    pose_norm = specimen_normalization(pose, specimen_ids)

    # Histogram grid
    x_grid, y_grid = np.linspace(0., 1., 11), np.linspace(0., 1., 11)

    # Compute histogram
    h = np.zeros((x_grid.size - 1, y_grid.size - 1))
    for xi in range(x_grid.size - 1):
        for yi in range(y_grid.size - 1):
            x_t = (x_grid[xi] <= pose_norm[:, 0]) * (pose_norm[:, 0] < x_grid[xi + 1])
            y_t = (y_grid[yi] <= pose_norm[:, 1]) * (pose_norm[:, 1] < y_grid[yi + 1])
            h[xi, yi] = (x_t * y_t).sum()
    # print h

    # Compute probability distribution
    prob = h / h.sum()
    # print prob

    return prob


def pose_variability2(pose, specimen_ids):
    # Compute probability distribution
    prob = pose_histogram2(pose, specimen_ids)

    # Compute entropy
    entropy = sps.entropy(prob.reshape(-1), base=10.)
    return entropy


def pose_diff2(pose_db1, specimen_ids_db1, pose_db2, specimen_ids_db2):
    # Compute probability distribution
    prob1 = pose_histogram2(pose_db1, specimen_ids_db1)
    prob2 = pose_histogram2(pose_db2, specimen_ids_db2)

    # Add regularization prior
    eps = 0.01
    prob1 += eps
    prob1 /= prob1.sum()
    prob2 += eps
    prob2 /= prob2.sum()

    # Compute KL divergence
    kl_div = sps.entropy(prob1.reshape(-1), prob2.reshape(-1), base=10.)
    return kl_div


def to_one_hot(idx, num_class):
    if isinstance(idx, int):
        one_hot = np.zeros((num_class,))
        one_hot[idx] = 1.
        return one_hot


def eval_class_acc(preds, targets):
    _, pred_classes = torch.max(preds.data, 1)
    # print(type(pred_classes), type(targets))
    corrects = (pred_classes == targets.data).sum()
    return 1.0 * corrects / targets.size(0)


def get_pred_classes(preds):
    _, pred_classes = torch.max(preds.data, 1)
    return pred_classes


def get_pred_coordinates(output_coord):
    pred_coordinates = []
    for i in range(len(output_coord)):
        pred_maps = output_coord[i].cpu().data.numpy()
        temp = np.stack([np.unravel_index(p.argmax(), p.shape) for p in pred_maps])
        pred_coordinates.append(np.fliplr(temp)/48.)
    return pred_coordinates

def invert_batchgrouping(batch):
    try:
        inverted_batch = []
        for i in range(len(batch)):
            inverted_batch += [batch[i][j] for j in range(len(batch[i]))]
        return np.array(inverted_batch)
    except:
        return batch

def euclideanDistance(prediction, gtruthHead, gtruthTail):
    headEuclid, tailEuclid = [], []
    head, tail = 0, 1
    nSmpl = len(prediction)
    for i in range(nSmpl):
        headEuclid.append (euclidean (prediction[i][head], gtruthHead[i]))
        tailEuclid.append (euclidean (prediction[i][tail], gtruthTail[i]))
    headEuclid = np.asarray (headEuclid)
    tailEuclid = np.asarray (tailEuclid)
    histData = {'Head Distribution': headEuclid, 'Tail Distribution': tailEuclid}
    avgHeadEuclid = headEuclid.mean ()
    avgTailEuclid = tailEuclid.mean ()
    avgEuclid = np.array ((avgHeadEuclid, avgTailEuclid)).mean ()
    return {'Head Distance':avgHeadEuclid,
            'Tail Distance':avgTailEuclid,
            'Avg Distance':avgEuclid,
            'Distribution':histData}
