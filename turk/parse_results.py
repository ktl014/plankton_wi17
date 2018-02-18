import csv, json
import numpy as np


if __name__ == '__main__':
    print 'Loading turk results...'
    batch_results = list(csv.reader(open('results/Batch_3113691_batch_results.csv')))
    header, batch_results = batch_results[0], batch_results[1:]
    batch_results = [(rst[15], ann) for rst in batch_results for ann in json.loads(rst[48])]

    # Parse poses
    print 'Parsing turk results...'
    results, img2idx = [], {}
    for wid, ann in batch_results:
        img = '/'.join(ann['url'].split('/')[-2:])
        img_shape = img.split('-')[-3:-1]
        width, height = int(img_shape[0]), int(img_shape[1])
        p = [ann['head']['relative_x']*width if isinstance(ann['head'], dict) else np.NaN,
             ann['head']['relative_y']*height if isinstance(ann['head'], dict) else np.NaN,
             ann['tail']['relative_x']*width if isinstance(ann['tail'], dict) else np.NaN,
             ann['tail']['relative_y']*height if isinstance(ann['tail'], dict) else np.NaN]

        if img not in img2idx:
            img2idx[img] = len(results)
            results.append({'image': None, 'turk_id': [], 'pose': [], 'confidence': [], 'orientation': [], 'focus': []})
            node = results[-1]
        else:
            node = results[img2idx[img]]

        assert node['image'] is None or node['image'] == img
        node['image'] = img
        node['turk_id'].append(wid)
        node['pose'].append(p)
        node['confidence'].append(ann['confidence'])
        node['orientation'].append(ann['z-dir'])
        node['focus'].append(ann['focus'])
    print '  Found a total of {} annotations for {} images.'.format(sum([len(rst['turk_id']) for rst in results]), len(results))

    # Process poses
    print '   Computing median head and tail positions...'
    images = [node['image'] for node in results]
    poses_med = np.zeros((len(images), 4), dtype=np.float) * np.NaN
    rej_images = 0
    for idx in range(len(images)):
        img_ann = results[idx]
        pose = np.array(img_ann['pose'], dtype=np.float)

        # Remove out of frame annotations
        in_frame = [i for i, p in enumerate(img_ann['pose']) if np.isnan(p).sum() == 0]
        if len(in_frame) == 0:
            rej_images += 1
            print ' * ', rej_images, images[idx], 'No in frame annotations!'
            continue
        pose = pose[in_frame]

        # Remove not confident annotations
        img_conf = [img_ann['confidence'][i] for i in in_frame]
        conf = [i for i, conf in enumerate(img_conf) if conf != 'Not Confident']
        if len(conf) == 0:
            rej_images += 1
            print ' * ', rej_images, images[idx], 'No confident annotations!'
            continue
        pose = pose[conf]

        poses_med[idx] = np.median(pose, axis=0)


    # Remove poses with out of frame annotations
    idx = list(np.where(np.isnan(poses_med).sum(axis=1) > 0)[0])
    poses_med = np.delete(poses_med, idx, 0)
    images = [images[i] for i in range(len(images)) if i not in idx]
    specimen_ids = [img.split('/')[0] for img in images]
    print '   Found {} images with pose annotation across {} specimens.'.format(len(specimen_ids), len(set(specimen_ids)))

    # Save results
    np.save('results/raw_data.npy', {'data': results, 'img2idx': img2idx})
    np.save('results/pose.npy',
            {'images': images,
             'head_x': poses_med[:, 0],
             'head_y': poses_med[:, 1],
             'tail_x': poses_med[:, 2],
             'tail_y': poses_med[:, 3]})