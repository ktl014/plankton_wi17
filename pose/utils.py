import csv
import numpy as np


def read_csv(csv_file):
    csv_reader = csv.DictReader(csv_file)
    data = []
    for row in csv_reader:
        datum = eval(row['Answer.annotation_data'])
        for d in datum:
            d['img_file'] = 'images/' + d['url'].split('/')[-1]
            d['specimen'] = d['url'].split('/')[5]
            d['class'] = d['specimen'].split('_')[0]
            data.append(d)
    return data


def filter_results(results):
    """
    filter the results of head and tail coordinates
    :param results: a list of coordinates represented as dictionary
                    ex: [{head: {x:1, y:1}, tail: {x:12, y:21}, others:etc}, ...]
    :return: the best coordinates
    """
    results = [r for r in results]
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
    return results[0]
