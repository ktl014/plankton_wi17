from utils.data import read_csv, data_filter
from utils.constants import *
import os
import numpy as np
from torch.utils.data import Dataset
from skimage import io


class PlanktonDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, output_size=32):
        assert isinstance(output_size, (int, tuple))

        # TODO: train/val split
        self.data = data_filter(read_csv(csv_file))

        self.img_dir = img_dir
        self.transform = transform

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        elif isinstance(output_size, tuple):
            assert len(output_size) == 2
            self.output_size = output_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # TODO: change this to actual image file position
        img_name = os.path.join(self.img_dir,
                                self.data.loc[idx, IMG_NAME])

        image = io.imread(img_name)
        coordinates = self.data.loc[idx, COORDINATES]
        cls = self.data.loc[idx, CLASS]

        sample = {'image': image,
                  'coordinates': np.asarray(coordinates),
                  'cls': cls}

        if self.transform:
            sample = self.transform(sample)

        # TODO: map the coordinates to Gaussian belief map
        # ** alternatively can be done at training time **
        # head_map = coordinates_to_gaussian_map(coordinates[:2], image, self.output_size)
        # tail_map = coordinates_to_gaussian_map(coordinates[2:], image, self.output_size)
        # bg_map = ?
        # sample['target_map'] = np.asarray([head_map, tail_map])

        return sample


if __name__ == '__main__':
    from transform import Rescale, RandomHorizontalFlip, RandomVerticalFlip, ToTensor
    from utils.vis import show_arrow_batch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt

    img_dir = '../images'
    csv_filename = 'data/Updated_Batch_3084800_batch_results.csv'

    # # update old frame of results to new frame
    # new_df = update_old('Batch_3084800_batch_results.csv', img_dir)
    # new_df.to_csv('Updated_' + csv_filename)

    transformed_dataset = PlanktonDataset(csv_file=csv_filename,
                                          img_dir=img_dir,
                                          transform=transforms.Compose([
                                              Rescale((224, 224)),
                                              RandomHorizontalFlip(),
                                              RandomVerticalFlip(),
                                              ToTensor()
                                          ]))

    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['coordinates'].size(),
              sample_batched['cls'])

        if i_batch == 3:
            plt.figure()
            show_arrow_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break
