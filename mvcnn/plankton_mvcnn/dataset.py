import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.data import *
from utils.constants import *
from transform import *
import os
import numpy as np
from numpy.random import choice
import pandas as pd
from skimage import io


class PlanktonDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, amp=1., std=3., output_size=48):
        assert isinstance(output_size, (int, tuple))

        self.data = pd.read_csv(csv_file)

        self.img_dir = img_dir
        self.transform = transform
        self.amp = amp
        self.std = std

        self.level = FAMILY
        self.classes = sorted(self.data[self.level].unique())
        self.class_to_index = {cls: i for i, cls in enumerate(self.classes)}
        self.num_class = len(self.classes)
        self.class_weights = self.num_class * \
            1. / (self.data[self.level].value_counts(normalize=True).sort_index().values.astype(np.float32) ** 1)

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        elif isinstance(output_size, tuple):
            assert len(output_size) == 2
            self.output_size = output_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir,
                                self.data.loc[idx, IMG_NAME])

        image = io.imread(img_name).astype(np.float32) / 255.
        coordinates = np.asarray([self.data.loc[idx, c] for c in COORDINATES_LIST])
        cls = {c: self.data.loc[idx, c] for c in CLASS_LIST}
        target_map = get_belief_map(coordinates, self.output_size, self.amp, self.std)
        # class_one_hot = to_one_hot(self.class_to_index[cls[self.level]], self.num_class)

        sample = {'image_name': img_name,
                  'image': image,
                  'coordinates': coordinates,
                  'cls': cls,
                  'class_index': self.class_to_index[cls[self.level]],
                  'target_map': target_map}

        if self.transform:
            sample = self.transform(sample)

        return sample

class MultiViewPlanktonDataSet(Dataset):
    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def __init__(self, csv_file, img_dir, transform=None, amp=1., std=3., output_size=48, views=12):
        assert isinstance(output_size, (int, tuple))

        self.data = pd.read_csv(csv_file)

        self.img_dir = img_dir
        self.transform = transform
        self.amp = amp
        self.std = std
        self.views = views

        self.level = FAMILY
        self.classes = sorted(self.data[self.level].unique())
        self.class_to_index = {cls: i for i, cls in enumerate(self.classes)}
        self.num_class = len(self.classes)
        self.class_weights = self.num_class * \
            1. / (self.data[self.level].value_counts(normalize=True).sort_index().values.astype(np.float32) ** 1)
        
        self.partition = self.generatePartition()

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        elif isinstance(output_size, tuple):
            assert len(output_size) == 2
            self.output_size = output_size

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        orginal_views = self.partition[index]
        views = []

        for view in orginal_views:
            img_name = os.path.join(self.img_dir,
                                self.data.loc[view, IMG_NAME])
            image = io.imread(img_name).astype(np.float32) / 255.
            coordinates = np.asarray([self.data.loc[view, c] for c in COORDINATES_LIST])
            cls = {c: self.data.loc[view, c] for c in CLASS_LIST}
            target_map = get_belief_map(coordinates, self.output_size, self.amp, self.std)
            sample = {'image_name': img_name,
                  'image': image,
                  'coordinates': coordinates,
                  'cls': cls,
                  'class_index': self.class_to_index[cls[self.level]],
                  'target_map': target_map}
            
            if self.transform is not None:
                sample = self.transform(sample)
            views.append(sample['image'])
        target = self.class_to_index[self.data.loc[orginal_views[0], self.level]]
        sample = {'views': views,
                  'target': target,
                  'view_ids': orginal_views}
        return sample

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.partition)
    
    def generatePartition(self):
        orig = self.data
        specimens = orig['specimen_id'].unique()
        partition = []
        for spec in specimens:
            spec_view = orig.loc[orig['specimen_id'] == spec]
            iters = len(spec_view)/self.views
            for i in range(iters):
                subset = choice(spec_view.index,size=self.views,replace=False)
                partition.append(subset)
                spec_view = spec_view.drop(index=subset)
#             if spec_view.shape[0] > 0:
#                 partition.append(np.array(spec_view.index))
        return partition
    
class DatasetWrapper(object):
    def __init__(self, phase, csv_filename, img_dir, input_size,
                 batch_size, amp, std, output_size=48, multiview=False, shuffle=True, views=12):

        self.phase = phase
        self.csv_filename = csv_filename
        self.img_dir = img_dir
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.amp = amp
        self.std = std

        self.normalizer = Normalize([0.5, 0.5, 0.5], [1, 1, 1])  # ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.data_transform = {
            TRAIN: transforms.Compose([#RandomRotation(randomNormalAngle,[0.25]),
                                       Rescale(self.input_size),
                                       RandomHorizontalFlip(),
                                       RandomVerticalFlip(),
                                       ToTensor(),
                                       self.normalizer]),
            VALID: transforms.Compose([#RandomRotation(randomAngle,[0,2*np.pi]),
                                       Rescale(self.input_size),
                                       ToTensor(),
                                       self.normalizer]),
            TEST:  transforms.Compose([#RandomRotation(randomAngle,[0,2*np.pi]),
                                       Rescale(self.input_size),
                                       ToTensor(),
                                       self.normalizer])
        }
        
        if multiview:
            self.dataset = MultiViewPlanktonDataSet(csv_file=self.csv_filename,
                                       img_dir=self.img_dir,
                                       transform=self.data_transform[self.phase],
                                       amp=self.amp,
                                       std=self.std,
                                       output_size=self.output_size,
                                       views=views)
        else:
            self.dataset = PlanktonDataset(csv_file=self.csv_filename,
                                       img_dir=self.img_dir,
                                       transform=self.data_transform[self.phase],
                                       amp=self.amp,
                                       std=self.std,
                                       output_size=self.output_size)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=(phase == TRAIN), num_workers=4)
        self.dataset_size = len(self.dataset)

    def __len__(self):
        return self.dataset_size

    def get_output_size(self):
        return self.output_size

    def get_class_weights(self):
        return torch.from_numpy(self.dataset.class_weights)

    def get_class_index(self):
        return self.dataset.class_to_index

    @staticmethod
    def get_num_class(csv_filename):
        dataset = PlanktonDataset(csv_filename, '')
        return dataset.num_class


if __name__ == '__main__':
    from utils.vis import show_arrow_batch
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Qt5Agg')

    # img_dir = '/data5/Plankton_wi18/rawcolor_db2/images'
    # csv_filename = 'data/data_train.csv'
    #
    # # # update old frame of results to new frame
    # # new_df = update_old('Batch_3084800_batch_results.csv', img_dir)
    # # new_df.to_csv('Updated_' + csv_filename)
    #
    # transformed_dataset = PlanktonDataset(csv_file=csv_filename,
    #                                       img_dir=img_dir,
    #                                       transform=transforms.Compose([
    #                                           Rescale((224, 224)),
    #                                           RandomHorizontalFlip(),
    #                                           RandomVerticalFlip(),
    #                                           ToTensor()
    #                                       ]))
    #
    # dataloader = DataLoader(transformed_dataset, batch_size=4,
    #                         shuffle=True, num_workers=4)
    #
    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch, sample_batched['image'].size(),
    #           sample_batched['coordinates'].size(),
    #           sample_batched['cls'])
    #
    #     if i_batch == 3:
    #         plt.figure()
    #         show_arrow_batch(sample_batched)
    #         plt.axis('off')
    #         plt.ioff()
    #         plt.show()
    #         break
