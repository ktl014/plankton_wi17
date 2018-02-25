import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils
import torch


def show_arrow(image, coordinates, cls):
    if isinstance(image, (np.ndarray, list)):
        image = np.copy(image)
    elif isinstance(image, torch.FloatTensor):
        image = (image.numpy()).transpose((1, 2, 0)).copy()

    if isinstance(coordinates, torch.FloatTensor):
        coordinates = coordinates.numpy()

    height, width = image.shape[:2]
    head = (int(coordinates[0] * width), int(coordinates[1] * height))
    tail = (int(coordinates[2] * width), int(coordinates[3] * height))
    cv2.arrowedLine(image, tail, head, (1., 0., 0.), 3)
    plt.imshow(image)
    plt.axis('off')
    plt.title(cls)
    plt.pause(0.001)


def show_arrow_batch(sample_batched):
    images_batch, coordinates_batch, cls_batch = \
        sample_batched['image'], sample_batched['coordinates'], sample_batched['cls']
    batch_size = len(images_batch)
    im_h, im_w = images_batch.size(2), images_batch.size(3)

    grid = utils.make_grid(images_batch)
    grid = grid.numpy().transpose((1, 2, 0))
    grid = grid.copy()

    for i in range(batch_size):
        hx, hy, tx, ty = coordinates_batch[i].numpy()
        head, tail = (int(tx * im_w + i * im_w), int(ty * im_h)), \
                     (int(hx * im_w + i * im_w), int(hy * im_h))
        cv2.arrowedLine(grid, tail, head, (1., 0., 0.), 3)

    plt.imshow(grid)
    # plt.title('  '.join(cls_batch))


def show_image_batch(images_batch):
    grid = utils.make_grid(images_batch)
    grid = grid.numpy().transpose((1, 2, 0))
    grid = grid.copy()

    plt.imshow(grid)
