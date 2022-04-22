#!/usr/bin/python

import os
import numpy as np
import time
import random
import yaml
#import open3d as o3d
from plyfile import PlyData
from yaml import CLoader as Loader # way faster 
from data.base_dataset import BaseDataloader
from data.object_model import ObjectModel
from data.camera_model import CameraModel
from data.data_utils import *
import data.tf_numpy as tf_numpy

class BlackBirdObjectModel(ObjectModel):
    def __init__(self, root_dir, idx, index, name, is_sym, num_points):
        super(BlackBirdObjectModel, self).__init__(root_dir, idx, index, name, num_points)
        self.model_path = os.path.join(self.root_dir, 'models')
        self.is_sym = is_sym
        self.diameter = self._read_diameter()

    def NED2EDN(self, pts):
        tf = np.array([[0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [1, 0, 0, 0],
                       [0, 0, 0, 1]])

        return np.matmul(pts, tf[:3, :3].T) + tf[:3, 3:].T

    def EDN2NED(self, pts):
        tf = np.array([[0., 0., 1., 0.],
                      [1., 0., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 0., 1.]])

        return np.matmul(pts, tf[:3, :3].T) + tf[:3, 3:].T

    def MODEL2EDN(self, pts):
        tf = np.array([[-1., 0., 0., 0.],
                       [0., -1., 0., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., 1.]])

        return np.matmul(pts, tf[:3, :3].T) + tf[:3, 3:].T

    def model2NED(self, pts):
        xyz_m_EDN = self.MODEL2EDN(pts.copy())
        xyz_m_NED = self.EDN2NED(xyz_m_EDN)
        return xyz_m_NED

    def _read_ply_file(self, filename):
        #f = open(filename)
        #pcd = o3d.io.read_point_cloud(filename)
        #coords = np.array(pcd.points)
        # o3d.visualization.draw_geometries([pcd])
        ply = PlyData.read(filename)
        vertex = ply['vertex']
        pts = np.array([vertex[t] for t in ('x', 'y', 'z')])
        pts = pts.transpose()
        return pts 

    def _read_model_pts(self):
        '''
        get Polygon data of the model
        '''
        ply_file = os.path.join(self.model_path, self.name+'.ply')
        model_points = self._read_ply_file(ply_file)

        mpoints_NED = self.model2NED(model_points)
        
        return mpoints_NED

    def _read_diameter(self):
        '''
        get model diameter
        '''
        info_file = os.path.join(self.model_path, 'models_info.yml')
        model_info = yaml.load(open(info_file, 'r'), Loader=Loader)
        diameter = model_info[self.id]['diameter'] / 1000.0 # in meter

        return diameter

class BlackBird(BaseDataloader):
    def __init__(self, root_dir, split, voxel_size, num_points=500, 
        select_obj=None, do_augmentation=True, image_based=False, load_seg_pred=None,
        num_farthest_pts=8):
        super(BlackBird, self).__init__(split, root_dir, 
            img_height=768, img_width=1024, 
            do_augmentation=do_augmentation, image_based=image_based)

        self.dataset = 'BlackBird'
        self.voxel_size = voxel_size
        self.select_obj = select_obj
        self.load_seg_pred = load_seg_pred # the path to predicted segmentation
        self.unit_scale = 1.0 # 
        self.num_farthest_pts = num_farthest_pts
        self.obj_ids = [0]
        self.obj_dics = {
            0:  'drone',
        }
        
        self.paths, self.list_class = self.get_paths()
        self.gt_poses = self.load_gt_pose()
        
        self.num_points = num_points
        self.symmetric_obj_id = [] # id, 1, 2, 4, ...
        
        self.models = self.get_models()

        # camera parameters
        self.cam_model = CameraModel(cam_cx=1024//2, cam_cy=768//2, 
            cam_fx=665.108, cam_fy=665.108)
        print('found {} images for the {} dataset'.format(
            self.__len__(), self.split))

    def get_models(self):
        '''
        collect model informaiton
        '''
        models = {}
        for idx in range(len(self.obj_ids)):
            obj_index = idx # continuous: 0, 1, 2, ...
            obj_id = self.obj_ids[idx] # discontinuous: 1, 2, 4, ...
            is_sym = self.is_symmetric_obj(obj_index)



            # !!! ### MY_TODO, set num_points = 5000
            # model = LineMODObjectModel(self.root_dir, obj_id, obj_index, self.obj_dics[obj_id], is_sym, self.num_points)
            model = BlackBirdObjectModel(self.root_dir, 
                obj_id, obj_index, self.obj_dics[obj_id], is_sym, 12000)




            model.load_model(self.root_dir, unit_scale=self.unit_scale)
            models[obj_id] = model
        return models

    def get_paths(self):
        '''
        get data path of LineMOD dataset
        '''
        paths_rgb = []
        paths_depth = []
        paths_mask = []
        paths_pred_mask = []
        list_class = []


        list_file = open( os.path.join(self.root_dir, self.split+'.txt') )
        line = list_file.readline().strip()
        while line:
            timestamp = line[4:-4]
            filename = 'left'+timestamp+'.jpg'
            filename_depth = 'left'+timestamp+'.png'
            rgb_file = os.path.join(self.root_dir, 'Camera_Left_RGB', filename_depth)
            mask_file = os.path.join(self.root_dir, 'Camera_Left_Segmented', filename)
            depth_file = os.path.join(self.root_dir, 'Camera_Left_Depth', filename_depth)

            paths_rgb.append(rgb_file)
            paths_depth.append(depth_file)
            paths_mask.append(mask_file)
            list_class.append(timestamp) # class index of each item
            if self.load_seg_pred is not None:
                pred_mask_file = os.path.join(self.load_seg_pred, filename)
                paths_pred_mask.append(pred_mask_file)


            line = list_file.readline().strip()

        # assert len(paths_rgb)>0, "rgb images not found."
        paths = {
           "rgb" : paths_rgb, 
           "depth" : paths_depth,
           "gt_mask" : paths_mask,
           'pred_mask': paths_pred_mask # empty
           }

        return paths, list_class

    def load_gt_pose(self):
        '''
        get gt pose: timestamp, translation, rotation
        '''
       
        gt_file = os.path.join(self.root_dir, 'poses_quat.txt')
        gt_poses = np.loadtxt(gt_file, delimiter=',', 
            dtype={'names': ('t', 'tx', 'ty', 'tz', 'w', 'qx', 'qy', 'qz'),
            'formats': ('U19', np.float, np.float, np.float, 
                np.float, np.float, np.float, np.float)})
        pose_dict = {row[0]:[row[1], row[2], row[3], row[4], row[5], row[6], row[7]] for row in gt_poses}
        return pose_dict

    def NED2EDNPose(self, pose_ned):
        tf = np.array([[0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [1, 0, 0, 0],
                       [0, 0, 0, 1]])
        return tf @ pose_ned

    def get_gt_pose(self, index):
        # convert (translation, rotation) to pose matrix
        timestamp = self.list_class[index]
        gt_pose = self.gt_poses[timestamp]
        trans = np.array(gt_pose[0:3]) / self.unit_scale # in meter
        rot = np.array(gt_pose[3:7])
        # # rot = tf_numpy.euler_matrix(rot_vec[0], rot_vec[1], 
        # #     rot_vec[2], 'rzyx')[:3, :3]
        from scipy.spatial.transform import Rotation as R
    
        rotation = R.from_quat(np.array([rot[1], rot[2], rot[3], rot[0]]))
        rotation = rotation.as_dcm()
        pose = np.concatenate((rotation, trans[np.newaxis].T), 1) # 3 x 4 matrix

        p_H = np.concatenate((pose, np.array([[0, 0, 0, 1]])), 0)
        pose = np.linalg.inv(p_H)

        pose = self.NED2EDNPose(pose)
        return pose[:3, :]

    def blackbird_mask_read(self, mask_path):
        '''
        return:
            seg_mask: (H, W), the value of each pixel: 1 for object, 0 for background
        '''
        import sys
        #np.set_printoptions(threshold=sys.maxsize)
        seg_mask = np.array(Image.open(mask_path)).astype(np.int32)
        #print(np.unique(seg_mask.reshape(-1, seg_mask.shape[2]), axis=0))
        color = [255//2, 255, 255]
        r = 50

        a = seg_mask[:, :, 0] >= (color[0] - r)
        b = seg_mask[:, :, 0] <= (color[0] + r)  
        c = seg_mask[:, :, 1] >= (color[1] - r) 
        d = seg_mask[:, :, 1] <= (color[1])  
        e = seg_mask[:, :, 2] >= (color[2] - r) 
        f = seg_mask[:, :, 2] <= (color[2]) 

        drone_mask = a * b * c * d * e * f 

        #seg_mask = [np.all(seg_mask == c, axis=-1) for c in color]
        #print(seg_mask.max(), seg_mask.min())
        #print(seg_mask.shape)
        #seg_mask = np.asarray(seg_mask,np.int32)[:, :, 0]
        return drone_mask

    def __getraw__(self, index):
        r"""data read directly from dataset

        """
        rgb = rgb_read(self.paths['rgb'][index])
        depth = depth_read_uint16(self.paths['depth'][index])
        # print(depth.shape)
        # print('gt',self.paths['gt_mask'][index])
        gt_mask_raw = self.blackbird_mask_read(self.paths['gt_mask'][index])
        gt_mask = self.remove_invalid_label(gt_mask_raw, depth)

        # print(self.gt_poses)
        pose = self.get_gt_pose(index)

        # print(depth.max(), depth.min())
        obj = 0
        candidates = {
           'rgb': rgb,
           'depth': depth,
           'gt_mask': gt_mask, # ground truth segmentation
           'pose': pose,
           'model': self.models[obj],
        }
        if self.load_seg_pred:
            # print(self.paths['pred_mask'][index])
            pred_mask = drone_pred_mask_read(self.paths['pred_mask'][index])
            # print(np.count_nonzero(pred_mask))
            pred_mask = self.remove_invalid_label(pred_mask, depth)
            candidates['pred_mask'] = pred_mask
        return candidates
