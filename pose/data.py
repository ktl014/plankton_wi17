from collections import defaultdict
from utils import *
import cv2


if __name__ == '__main__':
    with open('Batch_3084800_batch_results.csv', 'r') as csv_file:
        data = read_csv(csv_file)
    # keys: ['width', 'head', 'z-dir', 'url', 'img_file', 'confidence', 'focus', 'tail', 'height']
    image_data = defaultdict(list)
    for d in data:
        img_name = d['url'].split('/')[-1]
        image_data[img_name].append(d)
    results = []
    for image in image_data:
        r = filter_results(image_data[image])
        if r is not None:
            results.append(r)
    for d in results:
        img = cv2.imread(d['img_file'])
        if d['head'] == 'out of frame' or d['tail'] == 'out of frame':
            continue
        head, tail = (d['head']['x'], d['head']['y']), (d['tail']['x'], d['tail']['y'])
        cv2.arrowedLine(img, tail, head, (0, 0, 255), 3)
        cv2.imwrite('results1/' + d['url'].split('/')[-1], img)
