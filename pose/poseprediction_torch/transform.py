import cv2
import torch
import numpy as np


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

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

        copy = {key: sample[key] for key in sample}
        copy['image'] = image

        return copy


class ToTensor(object):
    def __call__(self, sample):
        image_name, image, coordinates, cls, class_one_hot, target_map = \
            sample['image_name'], sample['image'], sample['coordinates'], sample['cls'], sample['target_map'], sample['class_one_hot']

        image = image.transpose((2, 0, 1))

        return {'image_name': image_name,
                'image': torch.from_numpy(image),
                'coordinates': torch.from_numpy(coordinates),
                'cls': cls,
                'class_one_hot': torch.from_numpy(class_one_hot),
                'target_map': torch.from_numpy(target_map)}


class RandomHorizontalFlip(object):
    def __call__(self, sample, flip_prob=0.5):
        image, coordinates, target_map = sample['image'], sample['coordinates'], sample['target_map']

        if np.random.rand() < flip_prob:
            image = image[:, ::-1, :].copy()
            coordinates = np.asarray([
                1 - coordinates[0], coordinates[1],
                1 - coordinates[2], coordinates[3]])
            target_map = target_map[:, :, ::-1].copy()

        copy = {key: sample[key] for key in sample}
        copy['image'], copy['coordinates'], copy['target_map'] = image, coordinates, target_map

        return copy


class RandomVerticalFlip(object):
    def __call__(self, sample, flip_prob=0.5):
        image, coordinates, target_map = sample['image'], sample['coordinates'], sample['target_map']

        if np.random.rand() < flip_prob:
            image = image[::-1, :, :].copy()
            coordinates = np.asarray([
                coordinates[0], 1 - coordinates[1],
                coordinates[2], 1 - coordinates[3]])
            target_map = target_map[:, ::-1, :].copy()

        copy = {key: sample[key] for key in sample}
        copy['image'], copy['coordinates'], copy['target_map'] = image, coordinates, target_map

        return copy


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized image.
        """
        image = sample['image']

        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)

        copy = {key: sample[key] for key in sample}
        copy['image'] = image

        return copy

    def recover(self, sample):
        image = sample['image'].clone()

        for t, m, s in zip(image, self.mean, self.std):
            t.add_(m).mul_(s)

        copy = {key: sample[key] for key in sample}
        copy['image'] = image

        return copy


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
