#!/usr/bin/python

import os
import numpy as np
import random
import time
import yaml
from yaml import CLoader as Loader # way faster 
#import open3d as o3d

from data.base_dataset import BaseDataloader
from data.object_model import ObjectModel
from data.camera_model import CameraModel
from data.data_utils import *
#from bop_toolkit_lib import inout



#from open3d.open3d.geometry import estimate_normals
def apply_transform(pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts

class LineMODObjectModel(ObjectModel):
    def __init__(self, root_dir, idx, index, name, is_sym, num_points):
        super(LineMODObjectModel, self).__init__(root_dir, idx, index, name, num_points)

        self.is_sym = is_sym
        self.diameter = self._read_diameter()

    def _read_ply_file(self, filename):
        f = open(filename)
        assert f.readline().strip() == "ply"
        f.readline()
        f.readline()
        N = int(f.readline().split()[-1])
        while f.readline().strip() != "end_header":
            continue
        pts = []
        for _ in range(N):
            pts.append(np.float32(f.readline().split()[:3]))
        return np.array(pts)

    def _read_model_pts(self):
        '''
        get Polygon data of the model
        '''
        item = '%02d' % self.id
        self.model_path = os.path.join(self.root_dir, 'models')
        ply_file = os.path.join(self.model_path, 'obj_'+item+'.ply')
        
        # model_points = self._read_ply_file(ply_file)
        model_points = np.loadtxt(os.path.join(self.model_path, 'obj_'+item+'.xyz'))
        
        return model_points

    def _read_diameter(self):
        '''
        get model diameter
        '''
        info_file = os.path.join(self.root_dir, 'models', 'models_info.yml')
        model_info = yaml.load(open(info_file, 'r'), Loader=Loader)
        diameter = model_info[self.id]['diameter'] / 1000.0 # in meter

        return diameter

class LineMODOcclusion(BaseDataloader):
    def __init__(self, root_dir, split, voxel_size, num_points=1500, 
        select_obj=None, do_augmentation=False, image_based=False, load_seg_pred=None,
        num_farthest_pts=8):
        super(LineMODOcclusion, self).__init__(split, root_dir, 
            img_height=480, img_width=640, 
            do_augmentation=do_augmentation, image_based=image_based)

        self.dataset = 'LineMODOcclusion'
        self.voxel_size = voxel_size
        self.obj_ids = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15] 
        self.obj_dics = {
            1:  'ape',
            2:  'benchwise',
            4:  'camera',
            5:  'can',
            6:  'cat',
            8:  'driller',
            9:  'duck',
            10: 'eggbox',
            11: 'glue',
            12: 'holepuncher',
            13: 'iron',
            14: 'lamp',
            15: 'phone'
        }
        # note that, in occlusion LineMOD, only 8 objects are evaluated
        if select_obj is None:
            self.select_obj = [1, 5, 6, 8, 9, 10, 11, 12]
        else:
            self.select_obj = select_obj
        
        self.load_seg_pred = load_seg_pred # the path to predicted segmentation
        #self.paths, self.list_class = self.get_paths_json()
        self.paths, self.list_class = self.get_paths()

        self.num_points = num_points
        self.symmetric_obj_id = [10, 11]
        self.depth_factor = 1000.0 # the unit of depth image is meter
        self.models = self.get_models()

        # camera projection model
        self.cam_model = CameraModel(cam_cx=325.26110, cam_cy=242.04899, 
            cam_fx=572.41140, cam_fy=573.57043)
        
        print('found {} images for the {} dataset'.format(
            self.__len__(), self.split))

    def get_paths_json(self):
        '''
        get data path of Occlusion LineMOD dataset
        '''
        paths_rgb = []
        paths_depth = []
        paths_mask = []
        paths_pred_mask = []
        paths_pose = []

        list_class = []
        # list_inclass_index = [] 
        targets = inout.load_json(os.path.join(self.root_dir, 'test_targets_bop19.json'))
        for target in targets:
            if self.select_obj is not None:
                if not target['obj_id'] in self.select_obj:
                    continue
            obj_id = target['obj_id']
            obj_name = self.obj_dics[obj_id]
            file_index = "{:05d}".format(target['im_id'])
            rgb_file = os.path.join(self.root_dir, 'RGB-D', 'rgb_noseg', 'color_'+file_index+'.png')
            depth_file = os.path.join(self.root_dir, 'RGB-D', 'depth_noseg', 'depth_'+file_index+'.png')
            pose_file = os.path.join(self.root_dir, 'poses', obj_name, 'info_'+file_index+'.txt')
            mask_file = os.path.join(self.root_dir, 'masks', obj_name, str(int(file_index)) + '.png')
            paths_rgb.append(rgb_file)
            paths_depth.append(depth_file)
            paths_mask.append(mask_file)
            paths_pose.append(pose_file)
            
            list_class.append(obj_id) # class id of each item

        assert len(paths_rgb)>0, "rgb images not found."

        paths = {
            "rgb" : paths_rgb,
            "depth" : paths_depth,
            "pose" : paths_pose,
            "gt_mask" : paths_mask,
            "pred_mask": paths_pred_mask
            }


        return paths, list_class

    def get_paths(self):
        '''
        get data path of LineMOD dataset
        '''
        paths_rgb = []
        paths_depth = []
        paths_mask = []
        paths_pred_mask = []
        paths_pose = []

        list_class = []
        # list_inclass_index = [] 

        for obj_id in self.obj_ids:
            if self.select_obj is not None:
                if not obj_id in self.select_obj:
                    continue
            obj_name = self.obj_dics[obj_id]

            if self.split == 'train':
                list_file = open(os.path.join(self.root_dir, obj_name+'_train.txt'))
            else:
                list_file = open(os.path.join(self.root_dir, obj_name+'_val.txt'))
            line = list_file.readline().strip()
            while line:
                file_index = line.split('/')[2].split('_')[1].split('.')[0]
                
                rgb_file = os.path.join(self.root_dir, line)
                depth_file = os.path.join(self.root_dir, 'RGB-D', 'depth_noseg', 'depth_'+file_index+'.png')
                pose_file = os.path.join(self.root_dir, 'poses', obj_name, 'info_'+file_index+'.txt')
                mask_file = os.path.join(self.root_dir, 'masks', obj_name, str(int(file_index)) + '.png')
                paths_rgb.append(rgb_file)
                paths_depth.append(depth_file)
                paths_mask.append(mask_file)
                paths_pose.append(pose_file)

                if self.load_seg_pred is not None:
                    # example path: '../checkpoints/seg_occlusion/images_test/masks'
                    pred_mask_file = os.path.join(self.load_seg_pred, obj_name, str(int(file_index)) + '.png')
                    paths_pred_mask.append(pred_mask_file)

                list_class.append(obj_id) # class id of each item
                # list_inclass_index.append(int(file_index)) # item obj_id within corresponding class
                line = list_file.readline().strip()

        assert len(paths_rgb)>0, "rgb images not found."

        num =-1#100 
        paths = {
            "rgb" : paths_rgb[:num],
            "depth" : paths_depth[:num],
            "pose" : paths_pose[:num],
            "gt_mask" : paths_mask[:num],
            "pred_mask": paths_pred_mask[:num]
            }

        
        return paths, list_class

    def get_models(self):
        '''
        collect model informaiton
        '''
        models = {}
        for idx in range(len(self.obj_ids)):
            obj_index = idx # continuous: 0, 1, 2, ...
            obj_id = self.obj_ids[idx] # discontinuous: 1, 2, 4, ...

            if self.select_obj is not None:
                if not obj_id in self.select_obj:
                    continue
            
            obj_name = self.obj_dics[obj_id]
            is_sym = self.is_symmetric_obj(obj_index)
            model = LineMODObjectModel(self.root_dir, obj_id, obj_index, obj_name, is_sym, self.num_points)
            model.load_model(self.root_dir, unit_scale=1.0)
            models[obj_id] = model
        return models

    @staticmethod
    def read_gt_pose(pose_path):
        with open(pose_path) as pose_info:
            lines = [line.rstrip() for line in pose_info.readlines()]
            if 'rotation:' not in lines:
                return np.array([])
            row = lines.index('rotation:') + 1
            rotation = np.loadtxt(lines[row:row + 3])
            translation = np.loadtxt(lines[row + 4:row + 5])
        return np.concatenate([rotation, np.reshape(translation, newshape=[3, 1])], axis=-1)
        
    # def get_mask_bbox(self, mask):
    #     '''return the bounding box containing mask
    #     '''
    #     index = np.nonzero(mask)
    #     try:
    #         rmin = min(index[0]) #row
    #     except:
    #         print(mask)
    #     rmax = max(index[0])
    #     cmin = min(index[1]) #col
    #     cmax = max(index[1])
    #     # return [cmin, rmin, cmax-cmin+1, rmax-rmin+1]
    #     return [rmin, rmax, cmin, cmax]

    def correct_pose(self, id, pose_gt):
        r""" coordinates of CAD points are different 
        between original and occlusion set
        """

        if id in [1, 5, 8, 10, 11]:
            T0 = np.array([[ 0, -1,  0,  0],
                       [ 0,  0,  1,  0],
                       [-1,  0,  0,  0],
                       [ 0,  0,  0,  1]])
        elif id in [9, 12]:
            T0 = np.array([[ 0, -1,  0,  0],
                       [ 0,  0,  1,  0],
                       [1,  0,  0,  0],
                       [ 0,  0,  0,  1]])
        elif id in [6]:
            T0 = np.array([[ 0, 1,  0,  0],
               [ 0,  0,  1,  0],
               [1,  0,  0,  0],
               [ 0,  0,  0,  1]])
        else:
            pass
        T1 = np.array([[1., 0., 0., 0],
                       [0., -1., 0., 0],
                       [0., 0., -1., 0],
                       [0, 0, 0, 1]])
        pose_gt = np.concatenate((pose_gt, np.array([[0,0,0,1]]) ), 0)
        
        pose_gt = T1 @ pose_gt #@ T0
        return pose_gt[:3, :]

    def to_bop_pose(self, id, pose):
        r""" coordinates of CAD points are different 
        between original and occlusion set
        """

        pose = np.concatenate((pose, np.array([[0,0,0,1]]) ), 0)
        if id in [1, 5, 8, 10, 11]:
            T0 = np.array([[ 0, -1,  0,  0],
                       [ 0,  0,  1,  0],
                       [-1,  0,  0,  0],
                       [ 0,  0,  0,  1]])
            
        elif id in [6, 9, 12]:
            T0 = np.array([[ 0,  1,  0,  0],
                           [ 0,  0,  1,  0],
                           [ 1,  0,  0,  0],
                           [ 0,  0,  0,  1]])
        else:
            pass
        pose = pose @ T0
        #pose[1, :] = -pose[1, :]
        #pose[2, :] = -pose[2, :]
        
        return pose[:3, :]


    @staticmethod
    def create_fpfh_feature(self, pts):
        pcd = o3d.PointCloud()
        pcd.points = o3d.Vector3dVector(pts)
        estimate_normals(pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=60))
        fpfh = o3d.registration.compute_fpfh_feature(pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.025, max_nn=60))
        # estimate_normals(pcd,
        #     o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        # fpfh = o3d.registration.compute_fpfh_feature(pcd,
        #     o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        return fpfh.data
    
    def __getraw__(self, index):

        rgb = rgb_read(self.paths['rgb'][index])
        depth = depth_read(self.paths['depth'][index], 
            depth_factor=self.depth_factor)
        gt_mask_raw = mask_read(self.paths['gt_mask'][index])
        gt_mask = self.remove_invalid_label(gt_mask_raw, depth)

        obj_id = self.list_class[index]
        pose = self.read_gt_pose(self.paths['pose'][index])
        pose = self.correct_pose(obj_id, pose) # object coordinates are messed up in the dataset
        #pose = self.to_bop_pose(obj_id, pose) 
        candidates = {
           'rgb': rgb,
           'depth': depth,
           'gt_mask': gt_mask,
           'pose': pose,
           'model': self.models[obj_id],
        }

        if self.load_seg_pred:
            pred_mask = mask_read(self.paths['pred_mask'][index])
            pred_mask = self.remove_invalid_label(pred_mask, depth)
            candidates['pred_mask'] = pred_mask
        return candidates
