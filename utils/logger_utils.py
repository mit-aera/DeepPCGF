"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
cmap = plt.cm.viridis
import os

def scale_to_255(x):
    # normalize input [0, 1] to [0, 255]
    return x * 255

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)
        
def _is_tensor_image(img):
    return torch.is_tensor(img) and (img.ndimension() in {2, 3})

def tensor_to_numpy(x):
    # input is C, H, W
    # output is H, W, C
    x = torch.squeeze(x)
    assert _is_tensor_image(x), "x is not tensor image: size = {}".format(x.size())
    if x.dim() == 3:
        return np.transpose(x.cpu().numpy(), (1, 2, 0))
    elif x.dim() == 2:
        return x.cpu().numpy()

def numpy_to_tensor(x):
    # input is H, W, C
    # output is C, H, W
    x = np.squeeze(x)
    assert _is_numpy_image(x), "x is not numpy ndarray."
    if x.ndim == 3:
        return torch.from_numpy(np.transpose(x, (2, 0, 1)))
    elif x.ndim == 2:
        return torch.from_numpy(x)

def colorize_depthmap(depth, d_min=None, d_max=None):
    ch = 1
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min + 1e-6)
    hwc = 255 * cmap(depth_relative)[:,:,:ch] # H, W, C
    return hwc

def colorize_depthmap_batch(x):
    ch = 1
    batch_size, height, width = x.size()
    colored = torch.zeros(batch_size, ch, height, width)
    for i in range(batch_size):
        x_np = tensor_to_numpy(x[i].detach())
        colored[i] = numpy_to_tensor(colorize_depthmap(x_np))
    return colored

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_PIL_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_PLT_Histogram(error_list, path, show=False):
    res = np.zeros((1))
    for item in error_list:
        error = item.cpu().numpy()
        res = np.concatenate((res, error), axis=0)
    _ = plt.hist(res, bins='auto')
    plt.title("Prediction error histogram with 'auto' bins")
    plt.savefig(path, dpi=100)
    if show:
        plt.show()


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def transform_points(points, pose, add_noise=False, noise_range=0.03):
    '''
    transform points 
    '''
    assert pose.shape == (3, 4)
    rot = pose[:, 0:3]
    trans = pose[:, 3]
    target = np.dot(points, rot.T)
    if add_noise:
        add_t = np.array([random.uniform(-noise_range, noise_range) for i in range(3)])
        target = np.add(target, trans + add_t)
    else:
        target = np.add(target, trans)

    return target

import numpy.matlib as npm

# Q is a Nx4 numpy matrix and contains the quaternions to average in the rows.
# The quaternions are arranged as (w,x,y,z), with w being the scalar
# The result will be the average quaternion of the input. Note that the signs
# of the output quaternion can be reversed, since q and -q describe the same orientation
def averageQuaternions(Q):
    # Number of quaternions to average
    M = Q.shape[0]
    A = npm.zeros(shape=(4,4))

    for i in range(0,M):
        q = Q[i,:]
        # multiply q with its transposed version q' and add A
        A = np.outer(q,q) + A

    # scale
    A = (1.0/M)*A
    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)
    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:,0].A1)