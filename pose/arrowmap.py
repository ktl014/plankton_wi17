import csv
import urllib
import cStringIO
from PIL import Image
import cv2
import numpy as np
import time


def read_csv(csv_file):
    csv_reader = csv.DictReader(csv_file)
    data = []
    for row in csv_reader:
        datum = eval(row['Answer.annotation_data'])
        for d in datum:
            d['img_file'] = 'images/' + d['url'].split('/')[-1]
            data.append(d)
    return data


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
    with open('Batch_3084800_batch_results.csv', 'r') as csv_file:
        data = read_csv(csv_file)
    st = time.time()
    for i, d in enumerate(data):
        img_file = cStringIO.StringIO(urllib.urlopen(d['url']).read())
        img = np.asarray(Image.open(img_file))[..., ::-1].copy()
        head, tail = (d['head']['x'], d['head']['y']), (d['tail']['x'], d['tail']['y'])
        for c in head + tail:
            if not isinstance(c, int):
                continue
        cv2.arrowedLine(img, tail, head, (0, 0, 255), 5)
        # cv2.imshow('img', img)
        cv2.imwrite('results/' + str(head) + ',' + str(tail) + '.png', img)
        # cv2.waitKey(0)
        if i % 10 == 0:
            print('%d remaining, ETA: %f' % (len(data) - i, (time.time() - st) / (i + 1) * (len(data) - i)))
    cv2.destroyAllWindows()


if __name__ == '__main__':
    download_img()
