import csv
import numpy as np
import cv2


def read_csv(csv_file):
    csv_reader = csv.DictReader(csv_file)
    data = []
    for row in csv_reader:
        datum = eval(row['Answer.annotation_data'])
        for d in datum:
            d['trueimg'] =  {'x':int(d['url'].split('-')[-3:][1]), 'y':int(d['url'].split('-')[-3:][0])}
            d['img_file'] = 'images/' + d['url'].split('/')[-1]
            d['specimen'] = d['url'].split('/')[5]
            d['class'] = d['specimen'].split('_')[0]
            
            # Conversion
            # True X, Y
            #img = cv2.imread(d['img_file'])
            if d['head'] == 'out of frame' or d['tail'] == 'out of frame':
                continue
            height, width = int(d['url'].split('-')[-3:][0]), int(d['url'].split('-')[-3:][1])
            scale = width / 200.0
            head = (int(d['head']['x'] * scale), int(d['head']['y'] * scale))
            tail = (int(d['tail']['x'] * scale), int(d['tail']['y'] * scale))
            d['true_head'] = head
            d['true_tail'] = tail
            
            # Angle & Radius
            d['radius'] = np.sqrt((d['true_head'][0] - d['true_tail'][0]) **2 + (d['true_head'][1] - d['true_tail'][1]) ** 2)
            d['theta'] = np.arctan2(d['true_head'][1] - d['true_tail'][1], d['true_head'][0] - d['true_tail'][0]) * 180 / np.pi
            data.append(d)
    return data

def preprocessing(csv_file):
    with open('Batch_3084800_batch_results.csv', 'r') as csv_file:
        data = read_csv(csv_file)
    for i, d in enumerate(data):
        img = cv2.imread(d['img_file'])
        if d['head'] == 'out of frame' or d['tail'] == 'out of frame':
            continue
        height, width = img.shape[0], img.shape[1]
        scale = width / 200.0
        head = (int(d['head']['x'] * scale), int(d['head']['y'] * scale))
        tail = (int(d['tail']['x'] * scale), int(d['tail']['y'] * scale))
        d['true_head'] = head
        d['true_tail'] = tail
    return data

def filter_results(results):
    """
    filter the results of head and tail coordinates
    :param results: a list of coordinates represented as dictionary
                    ex: [{head: {x:1, y:1}, tail: {x:12, y:21}, others:etc}, ...]
    :return: the best coordinates
    """
    results = [r for r in results if r['head'] != 'out of frame' and r['tail'] != 'out of frame']
    while len(results) > 1:
        avg_x_head = np.mean([r['head']['x'] for r in results])
        avg_y_head = np.mean([r['head']['y'] for r in results])
        avg_y_tail = np.mean([r['tail']['y'] for r in results])
        avg_x_tail = np.mean([r['tail']['x'] for r in results])
        dists = [(r['head']['x'] - avg_x_head) ** 2 +
                 (r['head']['y'] - avg_y_head) ** 2 +
                 (r['tail']['x'] - avg_x_tail) ** 2 +
                 (r['tail']['y'] - avg_y_tail) ** 2 for r in results]
        idx = int(np.argmax(dists))
        results.remove(results[idx])
    return results[0] if results else None
