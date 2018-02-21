import cv2
import torch
import numpy as np


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, coordinates, cls = sample['image'], sample['coordinates'], sample['cls']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = cv2.resize(image, (new_w, new_h))

        return {'image': image, 'coordinates': coordinates, 'cls': cls}


class ToTensor(object):
    def __call__(self, sample):
        image, coordinates, cls = sample['image'], sample['coordinates'], sample['cls']

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'coordinates': torch.from_numpy(coordinates),
                'cls': cls}


class RandomHorizontalFlip(object):
    def __call__(self, sample, flip_prob=0.5):
        image, coordinates, cls = sample['image'], sample['coordinates'], sample['cls']

        if np.random.rand() < flip_prob:
            image = image[:, ::-1, :].copy()
            coordinates = np.asarray([
                1 - coordinates[0], coordinates[1],
                1 - coordinates[2], coordinates[3]])

        return {'image': image, 'coordinates': coordinates, 'cls': cls}


class RandomVerticalFlip(object):
    def __call__(self, sample, flip_prob=0.5):
        image, coordinates, cls = sample['image'], sample['coordinates'], sample['cls']

        if np.random.rand() < flip_prob:
            image = image[::-1, :, :].copy()
            coordinates = np.asarray([
                coordinates[0], 1 - coordinates[1],
                coordinates[2], 1 - coordinates[3]])

        return {'image': image, 'coordinates': coordinates, 'cls': cls}


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, coordinates, cls = sample['image'], sample['coordinates'], sample['cls']

        image = image / self.std
        image -= self.mean

        return {'image': image, 'coordinates': coordinates, 'cls': cls}


class EmptyTransform(object):
    def __call__(self, sample):
        return sample


if __name__ == '__main__':
    from dataset import PlanktonDataset
    from utils.vis import show_arrow
    import matplotlib.pyplot as plt
    from torchvision import transforms


    def test(sample, tsfms):
        original = EmptyTransform()

        n = len(tsfms) + 1
        r = int(np.sqrt(n))
        c = np.ceil(1.0 * n / r)

        for i, tsfm in enumerate([original] + tsfms):
            transformed_sample = tsfm(sample)

            ax = plt.subplot(r, c, i + 1)
            show_arrow(**transformed_sample)
            ax.set_title(type(tsfm).__name__)


    img_dir = '../images'
    csv_filename = 'data/Updated_Batch_3084800_batch_results.csv'

    plankton_dataset = PlanktonDataset(csv_file=csv_filename,
                                       img_dir=img_dir)
    scale = Rescale((224, 224))
    hflip = RandomHorizontalFlip()
    vflip = RandomVerticalFlip()
    normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    composed = transforms.Compose([hflip, vflip, Rescale((224, 224))])

    plt.figure()
    test(plankton_dataset[2], [scale, hflip, vflip, composed])
    plt.show()
