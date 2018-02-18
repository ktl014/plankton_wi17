import numpy as np
import scipy.stats as sps
import glob


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


def plankton_labels():
    # Load all specimens
    specimen_list = ['_'.join(fn.split('/')[-2:]) for fn in glob.glob('/data4/plankton_wi17/plankton/images_orig/*/*')]

    # Load all labels
    specimen_labels = [l.split('\t') for l in open('specimen_taxonomy.txt').read().splitlines()[1:]]
    specimen_labels = {l[0]: l[1:] for l in specimen_labels if not l[0].startswith('google')}
    for spc in specimen_list:
        if spc not in specimen_labels:
            specimen_labels[spc] = ['Unknown'] * 3

    return specimen_labels


def plankton_taxonomy():
    specimen_labels = plankton_labels()

    # Build taxonomy
    root = {'name': 'root',
            'specimen_list': sorted(specimen_labels.keys()),
            'parent': None,
            'children': []}
    tax_nodes = [[] for _ in range(4)]
    tax_nodes[0].append(root)
    for i in range(3):
        for parent in tax_nodes[i]:
            classes = list(set([specimen_labels[spc][i] for spc in parent['specimen_list']]))
            for cls in classes:
                node = {'name': cls,
                        'specimen_list': [spc for spc in parent['specimen_list'] if specimen_labels[spc][i] == cls],
                        'parent': parent,
                        'children': []}
                parent['children'].append(node)
                tax_nodes[i + 1].append(node)
    return root