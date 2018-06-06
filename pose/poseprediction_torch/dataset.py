import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.data import get_belief_map, to_one_hot, randomAngle
from utils.constants import *
from transform import *
import os
import numpy as np
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

        self.level = GENUS
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


class DatasetWrapper(object):
    def __init__(self, phase, csv_filename, img_dir, input_size, output_size,
                 batch_size, amp, std, shuffle=True):

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
            TRAIN: transforms.Compose([RandomRotation(randomAngle,[0,2*np.pi]),
                                       Rescale(self.input_size),
                                       RandomHorizontalFlip(),
                                       RandomVerticalFlip(),
                                       ToTensor(),
                                       self.normalizer]),
            VALID: transforms.Compose([RandomRotation(randomAngle,[0,2*np.pi]),
                                       Rescale(self.input_size),
                                       ToTensor(),
                                       self.normalizer]),
            TEST:  transforms.Compose([RandomRotation(randomAngle,[0,2*np.pi]),
                                       Rescale(self.input_size),
                                       ToTensor(),
                                       self.normalizer])
        }

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
    from transform import Rescale, RandomHorizontalFlip, RandomVerticalFlip, ToTensor, RandomRotation
    from utils.vis import show_arrow_batch
    from torch.utils.data import DataLoader
    from torchvision import transforms
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
