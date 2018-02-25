from torch.utils.data import Dataset
from utils.data import get_belief_map
from utils.constants import *
import os
import numpy as np
import pandas as pd
from skimage import io


class PlanktonDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, output_size=48):
        assert isinstance(output_size, (int, tuple))

        self.data = pd.read_csv(csv_file)

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
        img_name = os.path.join(self.img_dir,
                                self.data.loc[idx, IMG_NAME])

        image = io.imread(img_name).astype(np.float32) / 255.
        coordinates = np.asarray([self.data.loc[idx, c] for c in COORDINATES_LIST])
        cls = [self.data.loc[idx, c] for c in CLASS_LIST]
        target_map = get_belief_map(coordinates, self.output_size)

        sample = {'image': image,
                  'coordinates': coordinates,
                  'cls': cls,
                  'target_map': target_map}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    from transform import Rescale, RandomHorizontalFlip, RandomVerticalFlip, ToTensor
    from utils.vis import show_arrow_batch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Qt5Agg')

    img_dir = '/data5/Plankton_wi18/rawcolor_db/images'
    csv_filename = 'data/data_train.csv'

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
