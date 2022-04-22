import torch
import torchvision
import torchvision.transforms as transforms
import skimage.transform
import numpy as np
from PIL import Image

import data.data_utils as dutils
try:
    import accimage
except ImportError:
    accimage = None


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

class Resize(object):
    def __init__(self, do_augmentation=True, size=(480, 640)):
        self.do_augmentation = do_augmentation
        self.transform_1st = transforms.Resize(size, 1)
        self.transform_2nd = transforms.Resize(size, 2)
        self.size = size

    def __call__(self, sample):
        # print('1', sample['gt_mask'].max(), sample['gt_mask'].min())
        if self.do_augmentation:
            if not _is_pil_image(sample['rgb']):
                rgb = Image.fromarray(sample['rgb'])
                depth = Image.fromarray(sample['depth'])
                mask = Image.fromarray(sample['gt_mask'])
            else:
                rgb = sample['rgb']
                depth = sample['depth']
                mask = sample['gt_mask']
            rgb_resized = self.transform_2nd(rgb)
            depth_resized = self.transform_1st(depth)
            mask_resized = self.transform_1st(mask)

            sample['rgb'] = np.array(rgb_resized)
            sample['depth'] = np.array(depth_resized)
            sample['gt_mask'] = np.array(mask_resized).astype(int)
            # print('2', sample['gt_mask'].max(), sample['gt_mask'].min())
        return sample

class ResizeImage(object):
    def __init__(self, train=True, size=(480, 640)):
        self.train = train
        self.transform = transforms.Resize(size)

    def __call__(self, sample):
        if self.train:
            rgb = sample['rgb']
            rgb = self.transform(left_image)
            sample['rgb'] = rgb
        return sample


class DoTest(object):
    def __call__(self, sample):
        new_sample = torch.stack((sample, torch.flip(sample, [2])))
        return new_sample

class NormalizeRGB:
    def __init__(self, do_augmentation=True):
        self.do_augmentation = do_augmentation
        self.normalization = torchvision.transforms.Normalize(
            # mean = [0, 0, 0],
            # std=[255.0, 255.0, 255.0]
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __call__(self, batch):
        if self.do_augmentation:
            if batch['rgb'] is not None:
                batch['rgb'] = self.normalization(batch['rgb'])
            if batch['rgb_masked'] is not None:
                batch['rgb_masked'] = self.normalization(batch['rgb_masked'])

        return batch

class ColorJitter:
    def __init__(self, do_augmentation, jitter_range=0.2, hue=0.05):
        self.do_augmentation = do_augmentation
        self.jitter_range = jitter_range
        self.hue = hue
        
    def __call__(self, batch):
       if self.do_augmentation:
           brightness = self.jitter_range#float(self.jitter_range * (2 * torch.rand(1) - 1) + 1)
           contrast = self.jitter_range#float(self.jitter_range * (2 * torch.rand(1) - 1) + 1)
           saturation = self.jitter_range#float(self.jitter_range * (2 * torch.rand(1) - 1) + 1)
           color_jitter = torchvision.transforms.ColorJitter(brightness, contrast, saturation, self.hue)

           rgb = batch['rgb_masked']

           if _is_numpy_image(rgb):
              pil = Image.fromarray(rgb)
              rgb = np.array(color_jitter(pil))
           else:
              rgb = color_jitter(rgb) if rgb is not None else None
           batch['rgb_masked'] = rgb 
       return batch

class AddNoise:
    def __init__(self, key, do_augmentation, noise_range=0.03):
        self.key = key
        self.do_augmentation = do_augmentation
        self.noise_range = noise_range

    def __call__(self, batch):
        if self.do_augmentation:
            add_t = np.array([np.random.uniform(-self.noise_range, \
                self.noise_range) for i in range(3)])
            batch[self.key] = np.add(batch[self.key], add_t)

        return batch

class HorizontalFlip(object):
    """Horizontally flip the given ``numpy.ndarray``.

    Args:
        do_flip (boolean): whether or not do horizontal flip.

    """

    def __init__(self, do_flip):
        self.do_flip = do_flip

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be flipped.

        Returns:
            img (numpy.ndarray (C x H x W)): flipped image.
        """
        if not(_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))

        if self.do_flip:
            return np.fliplr(img)
        else:
            return img


class PCDJitter:
    r"""add noise to xyz coords

    """
    def __init__(self, do_augmentation, mu=0, sigma=0.0002):
        self.do_augmentation = do_augmentation
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        xyz = sample['xyz']
        pose_gt = sample['pose_gt']
        if np.random.random() < 0.95 and self.do_augmentation:
            xyz += np.random.normal(self.mu, self.sigma, (xyz.shape[0], xyz.shape[1]))
        sample = {'xyz': xyz, 'pose_gt': pose_gt}
        return sample

class PCDRotate:
    r"""apply rotation augmentaiton
    """
    def __init__(self, do_augmentation, rotation_range=360):
        self.do_augmentation = do_augmentation
        self.randg = np.random.RandomState()
        self.rotation_range = rotation_range

    def __call__(self, sample):
        xyz = sample['xyz']
        pose_gt = sample['pose_gt']
        if self.do_augmentation:
            T = dutils.sample_random_trans(xyz, self.randg, self.rotation_range)
            xyz = dutils.apply_transform(xyz, T)
            pose_gt = T @ pose_gt
        sample = {'xyz': xyz, 'pose_gt': pose_gt}
        return sample
#
# Image and Point Cloud data augmentation
#

def image_transforms(mode='train', augment_parameters=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
                     do_augmentation=True, transformations=None,  size=(256, 512)):
    if mode == 'train':
        data_transform = torchvision.transforms.Compose([\
            Resize(do_augmentation=do_augmentation),
            #Resize(),
            # AddNoise('pts_model_cam', do_augmentation),
            # ToTensor(),
            #ColorJitter(do_augmentation),
            # NormalizeRGB(do_augmentation),
        ])
        return data_transform
    elif mode == 'test' or mode == 'val':
        data_transform = torchvision.transforms.Compose([
            Resize(do_augmentation=do_augmentation),
            # ToTensor(),
            # NormalizeRGB(do_augmentation),
        ])
        return data_transform
    elif mode == 'custom':
        data_transform = torchvision.transforms.Compose(transformations)
        return data_transform
    else:
        print('Wrong mode')


def pcd_transforms(mode='train', do_augmentation=True, rotation_range=360):
    if mode == 'train':
        pcd_transform = torchvision.transforms.Compose([
            #PCDJitter(do_augmentation),
	    AddNoise('xyz', do_augmentation, 0.005),
            PCDRotate(do_augmentation, rotation_range)
        ])
    else:
        pcd_transform = torchvision.transforms.Compose([
            PCDJitter(do_augmentation=False)
        ])
    return pcd_transform 