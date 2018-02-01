import urllib
import cStringIO
from PIL import Image
import cv2
import numpy as np
import time
from utils import *


def download_img():
    with open('Batch_3084800_batch_results.csv', 'r') as csv_file:
        data = read_csv(csv_file)
    st = time.time()
    for i, d in enumerate(data):
        img_file = cStringIO.StringIO(urllib.urlopen(d['url']).read())
        img = np.asarray(Image.open(img_file))[..., ::-1].copy()
        img_name = d['url'].split('/')[-1]
        cv2.imwrite('images/' + img_name, img)
        if i % 10 == 0:
            print('%d remaining, ETA: %f' % (len(data) - i, (time.time() - st) / (i + 1) * (len(data) - i)))


def main():
    with open('Batch_180520_batch_results.csv', 'r') as csv_file:
        data = read_csv(csv_file)
    st = time.time()
    for i, d in enumerate(data):
        img = cv2.imread(d['img_file'])
        if d['head'] == 'out of frame' or d['tail'] == 'out of frame':
            continue
        height, width = img.shape[0], img.shape[1]
        head = (int(d['head']['relative_x'] * width), int(d['head']['relative_y'] * height))
        tail = (int(d['tail']['relative_x'] * width), int(d['tail']['relative_y'] * height))
        cv2.arrowedLine(img, tail, head, (0, 0, 255), 3)
        cv2.imwrite('results/' + d['url'].split('/')[-1], img)
        # print('head: ' + str(d['head']))
        # print('tail: ' + str(d['tail']))
        # print(head, tail)
        # print(img.shape)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # if i % 1000 == 0:
        #     print('%d remaining, ETA: %f' % (len(data) - i, (time.time() - st) / (i + 1) * (len(data) - i)))
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
