import csv
import urllib
import cStringIO
from PIL import Image
import cv2
import numpy as np


def read_csv(csv_file):
    csv_reader = csv.DictReader(csv_file)
    data = []
    for row in csv_reader:
        data.extend(eval(row['Answer.annotation_data']))
    return data


if __name__ == '__main__':
    with open('Batch_3084800_batch_results.csv', 'r') as csv_file:
        data = read_csv(csv_file)
    for i, d in enumerate(data):
        img_file = cStringIO.StringIO(urllib.urlopen(d['url']).read())
        img = np.asarray(Image.open(img_file))[..., ::-1].copy()
        head, tail = (d['head']['x'], d['head']['y']), (d['tail']['x'], d['tail']['y'])
        cv2.arrowedLine(img, tail, head, (0, 0, 255), 5)
        # cv2.imshow('img', img)
        cv2.imwrite('results/' + str(head) + ',' + str(tail) + '.png', img)
        # cv2.waitKey(0)
        if i % 100 == 0:
            print(i)
    cv2.destroyAllWindows()
