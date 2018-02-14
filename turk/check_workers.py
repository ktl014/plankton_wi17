import csv
import json
import random
import matplotlib.pyplot as plt

DB_DIR = '/data5/Plankton_wi18/rawcolor_db'

def plot_hit(hit_results):
    hit = json.loads(hit_results[48])
    f, axarray = plt.subplots(2, 5, figsize=(18, 9))
    for i in range(10):
        ann = hit[i]
        img_fn = '/'.join(ann['url'].split('/')[-2:])
        img = plt.imread(DB_DIR+'/images/' + img_fn)

        # print 'Image:      ', img_fn
        # print 'Confidence: ', ann['confidence']
        # print 'Focus:      ', ann['focus']
        # print 'Z-Dir:      ', ann['z-dir']
        ann_str = 'Confidence: {}\nFocus:      {}\nZ-Dir:      {}'.format(
            ann['confidence'], ann['focus'], ann['z-dir'])
        axarray[i%2, i/2].imshow(img)
        axarray[i % 2, i / 2].set_axis_off()
        if isinstance(ann['head'], dict) and isinstance(ann['tail'], dict):
            head_x = int(ann['head']['relative_x'] * img.shape[1])
            head_y = int(ann['head']['relative_y'] * img.shape[0])
            tail_x = int(ann['tail']['relative_x'] * img.shape[1])
            tail_y = int(ann['tail']['relative_y'] * img.shape[0])
            axarray[i % 2, i / 2].arrow(tail_x, tail_y, head_x - tail_x, head_y - tail_y,
                                        head_width=10, head_length=15, fc='r', ec='r')
        else:
            ann_str += '\nHead:       {}\nTail:       {}'.format(ann['head'], ann['tail'])
            # print 'Head:       ', ann['head']
            # print 'Tail:       ', ann['tail']
        axarray[i % 2, i / 2].text(5, -20, ann_str, color='white',
                                   horizontalalignment='left',
                                   verticalalignment='top',
                                   fontsize=10,
                                   bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 5})


if __name__ == '__main__':
    batch_rslt = list(csv.reader(open('results/Batch1_3113691.csv')))
    header, batch_rslt = batch_rslt[0], batch_rslt[1:]
    all_workers = list(set([rst[15] for rst in batch_rslt]))
    random.shuffle(all_workers)
    # all_workers = ['A31ZW5G443HRZS','AXVPMYCDTJ271','A1QOCVKM6YP6R4','A346O2IB1SKYLN','A361JAJA5MOSIU','A18HC62T1SROBJ']
    for i, wid in enumerate(all_workers):
        worker_rslt = [rst for rst in batch_rslt if rst[15]==wid]
        print '='*20, '{} of {}'.format(i, len(all_workers)), '='*20
        print 'Worker ID: ', wid
        print 'Num HITs:  ', len(worker_rslt)
        print ''

        random.shuffle(worker_rslt)
        for rst in worker_rslt[:5]:
            plot_hit(rst)
        plt.show()