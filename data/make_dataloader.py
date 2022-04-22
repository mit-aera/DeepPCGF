import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# from options.train_options import TrainOptions
from data.linemod_dataset import LineMOD
from data.linemod_occlusion_dataset import LineMODOcclusion
from data.blackbird_dataset import BlackBird
import data.data_utils as dutils
import torch



def collate_pair_fn_pcd(list_data):
    r"""
    0: model points
    1: sensor points
    """
    xyz0, coords0, feats0, \
        xyz1, coords1, feats1, \
        pts_farthest_m, pts_farthest_c, trans, \
        models, rgb, depth = list(zip(*list_data))
  
    xyz_batch0, coords_batch0, feats_batch0 = [], [], []
    xyz_batch1, coords_batch1, feats_batch1 = [], [], []
    # seg_mask_batch = []

    model_batch = []
    pts_farthest_m_batch0, pts_farthest_c_batch0 = [], [] 
    gt_dir_vecs_batch0, trans_batch = [], []
    len_batch0, len_batch1, len_batch2 = [], [], []

    rgb_batch = []
    depth_batch = []

    batch_id = 0
    for batch_id, _ in enumerate(coords0):
        N0 = coords0[batch_id].shape[0]
        N1 = coords1[batch_id].shape[0]
        # N2 = coords2[batch_id].shape[0]
        xyz_batch0.append(torch.from_numpy(xyz0[batch_id]))
        xyz_batch1.append(torch.from_numpy(xyz1[batch_id]))

        coords_batch0.append( torch.cat((torch.from_numpy(
                coords0[batch_id]).int(), torch.ones(N0, 1).int() * batch_id), 1))
        coords_batch1.append( torch.cat((torch.from_numpy(
                coords1[batch_id]).int(), torch.ones(N1, 1).int() * batch_id), 1))
        # coords_batch2.append( torch.cat((torch.from_numpy(
        #         coords2[batch_id]).int(), torch.ones(N2, 1).int() * batch_id), 1))

        feats_batch0.append(torch.from_numpy(feats0[batch_id]))
        feats_batch1.append(torch.from_numpy(feats1[batch_id]))
        # feats_batch2.append(torch.from_numpy(feats2[batch_id]))

        # seg_mask_batch.append(torch.from_numpy(seg_mask[batch_id]))

        len_batch0.append(N0)
        len_batch1.append(N1)
        # len_batch2.append(N2)

        model_batch.append(models[batch_id])

        pts_farthest_m_batch0.append(
            torch.from_numpy(pts_farthest_m[batch_id]).float() )
        pts_farthest_c_batch0.append(
            torch.from_numpy(pts_farthest_c[batch_id]).float() )

        trans_batch.append(torch.from_numpy(trans[batch_id][None, :]))
        
        # gt_dir_vecs_batch0.append(
        #     torch.from_numpy(gt_dir_vecs_c[batch_id]) )

        rgb_batch.append(rgb[batch_id])
        depth_batch.append(depth[batch_id])

    # Concatenate all lists
    xyz_batch0 = torch.cat(xyz_batch0, 0).float()
    coords_batch0 = torch.cat(coords_batch0, 0).int()
    feats_batch0 = torch.cat(feats_batch0, 0).float()

    xyz_batch1 = torch.cat(xyz_batch1, 0).float()
    coords_batch1 = torch.cat(coords_batch1, 0).int()
    feats_batch1 = torch.cat(feats_batch1, 0).float()

    # xyz_batch2 = torch.cat(xyz_batch2, 0).float()
    # coords_batch2 = torch.cat(coords_batch2, 0).int()
    # feats_batch2 = torch.cat(feats_batch2, 0).float()
    # seg_mask_batch = torch.cat(seg_mask_batch, 0).long()
 
    trans_batch = torch.cat(trans_batch, 0).float()
    # gt_dir_vecs_batch0 = torch.cat(gt_dir_vecs_batch0, 0).float()

    return {
      'xyz_m': xyz_batch0,
      'coords_m': coords_batch0,
      'feats_m': feats_batch0,
      'xyz_s': xyz_batch1,
      'coords_s': coords_batch1,
      'feats_s': feats_batch1,
      # 'xyz_d': xyz_batch2,
      # 'coords_d': coords_batch2,
      # 'feats_d': feats_batch2,
      # 'seg_mask': seg_mask_batch,
      'len_m': len_batch0,
      'len_s': len_batch1,
      # 'len_d': len_batch2,
      'model': model_batch,
      'pts_farthest_m': pts_farthest_m_batch0,
      'pts_farthest_c': pts_farthest_c_batch0,
      # 'gt_dir_vecs_c': gt_dir_vecs_batch0,
      'rgb': rgb_batch,
      'depth': depth_batch,
      'T_gt': trans_batch,
    }

def collate_pair_fn_rgb(list_data):
    r"""
    0: model points
    1: sensor points
    """
    rgb, depth, seg_mask, models, gt_pose = list(zip(*list_data))
  
    rgb_batch, depth_batch, seg_mask_batch = [], [], []

    model_batch = []
    pose_batch = []
    len_batch0 = []

    batch_id = 0
    for batch_id, _ in enumerate(rgb):
        N0 = rgb[batch_id].shape[0]

        seg_mask_batch.append(dutils.to_tensor(seg_mask[batch_id]))

        len_batch0.append([N0])

        model_batch.append(models[batch_id])
        pose_batch.append(torch.from_numpy(gt_pose[batch_id][None, :]))

        rgb_batch.append(dutils.rgb_to_4d_tensor(rgb[batch_id]))
        depth_batch.append(dutils.to_tensor(depth[batch_id])[:, None, :, :])

    # Concatenate all lists
    seg_mask_batch = torch.cat(seg_mask_batch, 0).long()
    rgb_batch = torch.cat(rgb_batch, 0).float()
    depth_batch = torch.cat(depth_batch, 0).float()
    pose_batch = torch.cat(pose_batch, 0).float()

    return {
      'gt_mask': seg_mask_batch,
      'rgb': rgb_batch,
      'depth': depth_batch, 
      'len': len_batch0,
      'model': model_batch,
      'T_gt': pose_batch,
    }
    

ALL_DATASETS = [LineMOD, LineMODOcclusion, BlackBird]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}

def make_dataloader(dataset, data_path, phase, batch_size, voxel_size, num_points, 
  num_threads=0, shuffle=None, select_obj=None, do_augmentation=True, image_based=False,
  load_seg_pred=None):
    assert phase in ['train', 'test']

    if select_obj is not None:
        select_obj = [int(item) for item in select_obj.split(',')]
    else:
        select_obj=None

    if shuffle is None:
        shuffle = phase != 'test'

    if dataset not in dataset_str_mapping.keys():
        raise ValueError(f'Dataset {dataset}, does not exists in ' +
                      ', '.join(dataset_str_mapping.keys()))

    Dataset = dataset_str_mapping[dataset]
    dset = Dataset(data_path, phase, voxel_size, num_points, 
      select_obj=select_obj, do_augmentation=do_augmentation, image_based=image_based,
      load_seg_pred=load_seg_pred) 


    if image_based:
      loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_threads,
        collate_fn=collate_pair_fn_rgb,
        pin_memory=True,
        drop_last=True)
    else:
      loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_threads,
        collate_fn=collate_pair_fn_pcd,
        pin_memory=True,
        drop_last=True)
    return loader



if __name__ == '__main__':
    voxel_size=0.002
    loader = make_dataloader(dataset='LineMOD', 
      data_path='../data/Linemod_preprocessed', 
      phase='test', batch_size=1, voxel_size=voxel_size, 
      num_points=5000, num_threads=1, shuffle=False)
    data_loader_iter = loader.__iter__()
    input_dict = data_loader_iter.next()


    xyz0 = input_dict['xyz0'].numpy()
    xyz1 = input_dict['xyz1'].numpy()
    trans = input_dict['T_gt'][0].numpy()

    xyz0_transformed = dutils.apply_transform(xyz0, trans)

    pcd0 = dutils.make_open3d_point_cloud(xyz0_transformed)
    pcd1 = dutils.make_open3d_point_cloud(xyz1)
    # # find the density of point cloud

    # # find the overlap of two point clouds
    
    ratio = dutils.compute_overlap_ratio(pcd0, pcd1, voxel_size, 
      match_thresh_ratio=2.5)
    print(ratio)
    # Make point clouds using voxelized points
    # pcd = dutils.make_open3d_point_cloud(xyz)
    # Select features and points using the returned voxelized indices
    # pcd.colors = o3d.utility.Vector3dVector(color[sel])
    # pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points)[sel])

