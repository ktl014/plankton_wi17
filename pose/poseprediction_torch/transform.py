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
        image_name, image, coordinates, cls, class_index, target_map = \
            sample['image_name'], sample['image'], sample['coordinates'], sample['cls'], sample['class_index'], sample['target_map']

        image = image.transpose((2, 0, 1))

        return {'image_name': image_name,
                'image': torch.from_numpy(image),
                'coordinates': torch.from_numpy(coordinates),
                'cls': cls,
                'class_index': class_index,
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

class RandomRotation():
    def __init__(self,angleFx,params):
        self.angleFx = angleFx
        self.params = params
        
    def __call__(self, sample):
        image, coordinates, target_map = sample['image'], sample['coordinates'], sample['target_map']
        x_axis = np.array([1,0])
        
        h = image.shape[0]
        w = image.shape[1]
    
        head = np.array((int(coordinates[0] * w), int(coordinates[1] * h)))
        tail = np.array((int(coordinates[2] * w), int(coordinates[3] * h)))
        
        pose = np.array([head[0] - tail[0],head[1] - tail[1]])
        dot = np.dot(x_axis,pose)
        norm = np.linalg.norm(pose,axis=0)
        
        copy = {key: sample[key] for key in sample}
        if norm == 0:
            image = image.copy()
            coordinates = coordinates.copy()
            target_map = target_map.copy()
            
            copy['image'], copy['coordinates'], copy['target_map'] = image, coordinates, target_map
        else:
            angle = np.arccos(np.divide(dot,norm))

            if pose[1] < 0:
                angle = 2*np.pi - angle

            desired_ang = self.angleFx(*self.params)
            rotation_ang = desired_ang - angle
            center = np.vstack(((head[0] + tail[0])/2,(head[1] + tail[1])/2))

            M = cv2.getRotationMatrix2D((center[0],center[1]),np.rad2deg(-rotation_ang),1)
            bounds = np.array([[0,0,1],[w,0,1],[0,h,1],[w,h,1]]).T
            bounds_rot = M.dot(bounds)
            min_x = min(bounds_rot[0,:])
            max_x = max(bounds_rot[0,:])
            min_y = min(bounds_rot[1,:])
            max_y = max(bounds_rot[1,:])    

            h_new = max_y - min_y
            w_new = max_x - min_x
            tx = np.abs(np.min([min_x,0]))
            ty = np.abs(np.min([min_y,0]))

            tx -= np.abs(w_new - np.max([max_x,w_new]))
            ty -= np.abs(h_new - np.max([max_y,h_new]))

            bounds_rot = bounds_rot + [[tx],[ty]]
            cosM = np.abs(M[0, 0])
            sinM = np.abs(M[0, 1])

            M[0, 2] += tx
            M[1, 2] += ty

            dst = cv2.warpAffine(image,M,(int(np.ceil(w_new)),int(np.ceil(h_new))))

            head_rot = M.dot(np.vstack((head.reshape((2,1)),[1])))
            tail_rot = M.dot(np.vstack((tail.reshape((2,1)),[1])))
            corrdinates_rot = [head_rot[0][0]/float(dst.shape[1]),head_rot[1][0]/float(dst.shape[0]),tail_rot[0][0]/float(dst.shape[1]),tail_rot[1][0]/float(dst.shape[0])]

            image = dst
            coordinates = corrdinates_rot
            target_map = get_belief_map(coordinates, target_map.shape , 1., 3.)

            
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
