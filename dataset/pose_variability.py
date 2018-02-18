import utils
import numpy as np
from matplotlib import pyplot as plt
import random

LEVEL = 'DATASET'   # SPECIMEN GENUS FAMILY ORDER DATASET
NUM_COLS = 2        # Number of columns for


if __name__ == '__main__':
    print 'Loading pose dataset.'
    poses = np.load('/data5/morgado/projects/plankton/turk/results/pose.npy').all()
    specimen_ids = [img.split('/')[0] for img in poses['images']]
    head_x, head_y, tail_x, tail_y = poses['head_x'], poses['head_y'], poses['tail_x'], poses['tail_y']
    plankton_labels = utils.plankton_labels()

    if LEVEL == 'SPECIMEN':
        specimen_sets = {spc: [spc] for spc in set(specimen_ids)}
    elif LEVEL == 'GENUS':
        genus = [plankton_labels[spc][2] for spc in specimen_ids]
        specimen_sets = {cls: [spc for spc in specimen_ids if plankton_labels[spc][2]==cls] for cls in set(genus)}
    elif LEVEL == 'FAMILY':
        family = [plankton_labels[spc][1] for spc in specimen_ids]
        specimen_sets = {cls: [spc for spc in specimen_ids if plankton_labels[spc][1]==cls] for cls in set(family)}
    elif LEVEL == 'ORDER':
        family = [plankton_labels[spc][0] for spc in specimen_ids]
        specimen_sets = {cls: [spc for spc in specimen_ids if plankton_labels[spc][0]==cls] for cls in set(family)}
    elif LEVEL == 'DATASET':
        specimen_sets = {'Dataset': specimen_ids}

    print 'Pose variability vs {}.'.format(LEVEL)
    class_var, class_idx = {}, {}
    for cls in specimen_sets:
        idx = [i for i, spc in enumerate(specimen_ids) if spc in specimen_sets[cls]]
        class_var[cls] = utils.pose_variability(head_x[idx], head_y[idx], tail_x[idx], tail_y[idx], [specimen_ids[i] for i in idx])
        class_idx[cls] = idx

    # Show results
    classes = class_var.keys()
    order = np.argsort([class_var[cls] for cls in classes])
    classes = [classes[i] for i in order]

    fig, axarr = plt.subplots(max(len(classes)/NUM_COLS+1, 2), NUM_COLS)
    for i in range(len(axarr)):
        for j in range(len(axarr[0])):
            axarr[i, j].set_axis_off()
    for i_ax, cls in enumerate(classes):
        print '  ', cls, class_var[cls], len(class_idx[cls])
        # axarr[i_ax / NUM_COLS, i_ax % NUM_COLS].set_title('{:.03f}'.format(class_var[cls]))
        axarr[i_ax / NUM_COLS, i_ax % NUM_COLS].set_title('{:.03f} ({})'.format(class_var[cls], cls.split()[0]))
        axarr[i_ax / NUM_COLS, i_ax % NUM_COLS].plot([0, 0], [-1, 1], 'r')
        axarr[i_ax / NUM_COLS, i_ax % NUM_COLS].plot([-1, 1], [0, 0], 'r')
        axarr[i_ax / NUM_COLS, i_ax % NUM_COLS].plot(np.cos(np.arange(0, 2.1*np.pi, 0.1)), np.sin(np.arange(0, 2.1*np.pi, 0.1)), 'k')

        idx = class_idx[cls]
        _, pose_lng, pose_lat = utils.estimate_sph_pose(
            head_x[idx] - tail_x[idx], head_y[idx] - tail_y[idx], [specimen_ids[i] for i in idx])
        pose = np.stack(utils.sph2xyz(np.ones_like(pose_lng), pose_lng, pose_lat), axis=1)
        random.shuffle(pose)
        for p in pose[:200]:
            axarr[i_ax / NUM_COLS, i_ax % NUM_COLS].plot(p[0], p[1], '.b')
        axarr[i_ax / NUM_COLS, i_ax % NUM_COLS].set_xlim(-1.1, 1.1)
        axarr[i_ax / NUM_COLS, i_ax % NUM_COLS].set_ylim(-1.1, 1.1)
    plt.show()
