from collections import defaultdict
from utils import *


if __name__ == '__main__':
    with open('Batch_3084800_batch_results.csv', 'r') as csv_file:
        data = read_csv(csv_file)
    # keys: ['width', 'head', 'z-dir', 'url', 'img_file', 'confidence', 'focus', 'tail', 'height']
    image_data = defaultdict(list)
    for d in data:
        img_name = d['url'].split('/')[-1]
        image_data[img_name].append(d)


