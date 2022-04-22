import os
import numpy as np
#import open3d as o3d
import random
import copy
from scipy.linalg import expm, norm
# from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from PIL import Image
import torch
from data.tf_torch import transform_points



# ---------------------------
# Data IO
# ---------------------------

def rgb_read(filename):
    rgb_image = np.array(Image.open(filename))[:, :, :3] #/ 255.0
    # rgb_image = plt.imread(filename) #/ 255.0
    return rgb_image

def back_read(filename):
    return np.array(Image.open(filename).convert("RGB")) / 255.0

#### depth image io #####
def depth_read(filename, depth_factor=1000.0):
    depth_image = np.array(Image.open(filename)) / depth_factor # convert to meter
    return depth_image

def depth_read_uint16(filename):
    # depth range (0, 100m)
    depth_image = np.array(Image.open(filename)) / (2**16) * 100
    
    return depth_image
def show_depth_image(depth):
    depth = depth#*5.12#*5.12
    im = Image.fromarray(depth)#.convert("RGB")
    im.show()

def save_depth_as_uint16(depth, filename=None, max_depth=100.0):
    a = depth * (2**16) / max_depth
    depth = Image.fromarray(a.astype(np.uint16))
    if filename is None:
        depth.save('depth.png')
    else:
        depth.save(filename)
#### depth image io #####

##### mask operation #####
def mask_read(mask_path):
    '''
    return:
        seg_mask: (H, W), the value of each pixel: 1 for object, 0 for background
    '''
    seg_mask = np.array(Image.open(mask_path)).astype(np.int32)
    # print(np.count_nonzero(seg_mask))
    if len(seg_mask.shape) == 3:
        seg_mask=np.sum(seg_mask, 2)>0
    elif len(seg_mask.shape) == 2:
        seg_mask=seg_mask>0
    else:
        raise NotImplementedError('unrecognized data shape')
    seg_mask=np.asarray(seg_mask,np.int32)

    return seg_mask

def drone_pred_mask_read(mask_path):
    '''
    return:
        seg_mask: (H, W), the value of each pixel: 1 for object, 0 for background
    '''
    seg_mask = np.array(Image.open(mask_path)).astype(np.uint8)
    seg_mask=seg_mask>0
    seg_mask=np.asarray(seg_mask,np.uint8)

    return seg_mask

def convert_mask_to_image(mask):
    mask = mask*255
    im = Image.fromarray(mask.astype(np.uint8))#.convert("RGB")
    return im

def show_mask_image(mask):
    r"""numpy_array contains 0 for background, 1 for object
    """
    im = convert_mask_to_image(mask)
    im.show()

def save_mask_as_gray(mask, filename=None):
    r"""
    args: 
      mask: 0 for background, 1 for foreground
    """
    im = convert_mask_to_image(mask)
    if filename is None:
        im.save('mask.png')
    else:
        im.save(filename)
#### mask operation #####

# ---------------------------
# End of Data IO
# ---------------------------


# ---------------------------
# Point Cloud Sampling
# ---------------------------

def sample_random_trans(pcd, randg, rotation_range=360):
  T = np.eye(4)
  R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
  T[:3, :3] = R
  T[:3, 3] = R.dot(-np.mean(pcd, axis=0)) # move the mean to the origin
  # T[:3, 3] = 3*(np.random.rand(3) - 0.5)
  return T


def sample_3d_vector():
    theta = np.random.uniform(0, 2*np.pi)
    z = np.random.uniform(-1, 1)
    return np.array([np.sqrt(1-z*z)*np.cos(theta), np.sqrt(1-z*z)*np.sin(theta), z])

def sample_knn(pcd, K, ref_pt='center'):
    r"""Randomly sample K nearest neighbors to a point, from a given point cloud pcd
    args:
        ref_pt: the reference point. If ref_pt='center',
            we choose the center of the point cloud as the reference point.
            otherwise, we take the center point as starting point, randomly sample a
            direction vector in 3D, extend r=5m along the direction vector to obtain
            the reference point.
    """
    if ref_pt == 'center':
        ref = np.mean(pcd, axis=0)
    else:
        direction = sample_3d_vector()
        r = 10
        ref = np.mean(pcd, axis=0) + r * direction
    neigh = NearestNeighbors(n_neighbors=K)
    neigh.fit(pcd)
    _, ind = neigh.kneighbors(np.array([ref]))
    return pcd[ind][0]
    
def sample_partial_pcd(pcd):
    r"""Randomly generate a plane which passes the center of a point cloud,
    return points that are on one side of the plane
    
    args:
        pcd: (num_ponts, 3), input point cloud
    """
    center = np.mean(pcd, axis=0)
    normal = sample_3d_vector()
    
    sign = np.dot(pcd - center, normal)    
    return pcd[sign > 0]

def uniform_sample_pcd(pcd, num_sample):
    '''downsample 3D points
    args: 
        num_points: number of points after sampling
    '''
    # np.random.seed(0)
    num_pts = pcd.shape[0]
    print(num_sample)
    dellist = [j for j in range(0, num_pts)]
    dellist = random.sample(dellist, num_pts - num_sample)
    # dellist = dellist[:len(self.raw_model_points)-num_points] # remove randomness, use for test
    downsampled_pcd = np.delete(pcd, dellist, axis=0)
    return downsampled_pcd

# ---------------------------
# End of Point Cloud Sampling
# ---------------------------

def rgb_to_3d_tensor(rgb):
    r"""convert rgb numpy array to tensor
    """
    return torch.from_numpy(rgb.transpose((2, 0, 1)).copy()).float()

def rgb_to_4d_tensor(rgb):
    r"""convert rgb numpy array to tensor
    """
    return rgb_to_3d_tensor(rgb)[None, :]

def to_tensor(np_array):
    return torch.from_numpy(np_array.copy())[None, :]

def apply_transform(pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts

def M(axis, theta):
  return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

def get_matching_indices(source, target, trans, search_voxel_size, K=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    source_copy.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)

    match_inds = []
    for i, point in enumerate(source_copy.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds

def get_matching_indices_no_trans(source, target, search_voxel_size, K=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    # source_copy.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)

    match_inds = []
    for i, point in enumerate(source_copy.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds

def compute_overlap_ratio(pcd0, pcd1, voxel_size, match_thresh_ratio=1):
    pcd0_down = pcd0.voxel_down_sample(voxel_size)
    pcd1_down = pcd1.voxel_down_sample(voxel_size)
    # print(np.asarray(pcd0_down.points).shape, np.asarray(pcd0.points).shape)
    # print('1',np.asarray(pcd1_down.points).shape, np.asarray(pcd1.points).shape)
    matching01 = get_matching_indices_no_trans(pcd0_down, pcd1_down, 
        match_thresh_ratio*voxel_size, 1)
    matching10 = get_matching_indices_no_trans(pcd1_down, pcd0_down,
        match_thresh_ratio*voxel_size, 1)
    overlap0 = len(matching01) / len(pcd0_down.points)
    overlap1 = len(matching10) / len(pcd1_down.points)
    print(overlap0, overlap1)
    return max(overlap0, overlap1)

def write_points(filename, pts, colors=None):
    has_color=pts.shape[1]>=6
    with open(filename, 'w') as f:
        for i,pt in enumerate(pts):
            if colors is None:
                if has_color:
                    f.write('{} {} {} {} {} {}\n'.format(
                        pt[0],pt[1],pt[2],int(pt[3]),int(pt[4]),int(pt[5])))
                else:
                    f.write('{} {} {}\n'.format(pt[0],pt[1],pt[2]))

            else:
                if colors.shape[0]==pts.shape[0]:
                    f.write('{} {} {} {} {} {}\n'.format(
                        pt[0],pt[1],pt[2],
                        int(colors[i,0]),
                        int(colors[i,1]),
                        int(colors[i,2])))
                else:
                    f.write('{} {} {} {} {} {}\n'.format(
                        pt[0],pt[1],pt[2],
                        int(colors[0]),
                        int(colors[1]),
                        int(colors[2])))
