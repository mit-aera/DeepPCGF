#!/usr/bin/python

import os
import numpy as np
import time
import random
import yaml
from yaml import CLoader as Loader # way faster 
#import open3d as o3d
from data.base_dataset import BaseDataloader
from data.object_model import ObjectModel
from data.camera_model import CameraModel
from data.data_utils import *
import data.tf_numpy as tf_numpy
#from bop_toolkit_lib import inout

import matplotlib.pyplot as plt

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
        model_points = self._read_ply_file(ply_file)
        
        return model_points

    def _read_diameter(self):
        '''
        get model diameter
        '''
        info_file = os.path.join(self.root_dir, 'models', 'models_info.yml')
        model_info = yaml.load(open(info_file, 'r'), Loader=Loader)
        diameter = model_info[self.id]['diameter'] / 1000.0 # in meter

        return diameter

class LineMOD(BaseDataloader):
    def __init__(self, root_dir, split, voxel_size, num_points=500, 
        select_obj=None, do_augmentation=True, image_based=False, load_seg_pred=None):
        super(LineMOD, self).__init__(split, root_dir, 
            img_height=480, img_width=640, 
            do_augmentation=do_augmentation, image_based=image_based)

        self.dataset = 'LineMOD'
        self.voxel_size = voxel_size
        self.select_obj = select_obj
        self.load_seg_pred = load_seg_pred # the path to predicted segmentation
        
        self.unit_scale = 1000.0 # convert unit from mm to m in CAD model, gt_pose
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

        self.paths, self.list_obj_ids, self.list_im_ids = self.get_paths()
        #self.paths, self.list_obj_ids, self.list_im_ids = self.get_paths_json()
        self.meta = self.get_meta()
        
        self.num_points = num_points
        self.symmetric_obj_id = [10, 11] # id, 1, 2, 4, ...
        
        self.models = self.get_models()

        # camera parameters
        self.cam_model = CameraModel(cam_cx=325.26110, cam_cy=242.04899, 
            cam_fx=572.41140, cam_fy=573.57043)
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
            model = LineMODObjectModel(self.root_dir, obj_id, obj_index, self.obj_dics[obj_id], is_sym, 5000)




            model.load_model(self.root_dir, unit_scale=self.unit_scale)
            models[obj_id] = model
        return models

    def get_paths_json(self):
        '''
        get data path of Occlusion LineMOD dataset
        '''
        paths_rgb = []
        paths_depth = []
        paths_mask = []
        paths_pred_mask = []
        paths_pose = []

        list_obj_ids = []
        list_im_ids = []
        targets = inout.load_json(os.path.join(self.root_dir, 'test_targets_bop19.json'))
        for target in targets:
            if self.select_obj is not None:
                if not target['obj_id'] in self.select_obj:
                    continue
            obj_id = target['obj_id']
            if not obj_id in self.obj_ids:
                continue
            obj_name = self.obj_dics[obj_id]
            file_index = "{:04d}".format(target['im_id'])
            obj_id_str = "{:02d}".format(obj_id)
            rgb_file = os.path.join(self.root_dir, 'data', obj_id_str, 'rgb', file_index+'.png')
            depth_file = os.path.join(self.root_dir, 'data', obj_id_str, 'depth', file_index+'.png')
            mask_file = os.path.join(self.root_dir, 'data', obj_id_str, 'mask', file_index+'.png')
            paths_rgb.append(rgb_file)
            paths_depth.append(depth_file)
            paths_mask.append(mask_file)

            list_obj_ids.append(obj_id) # obj id of each item
            list_im_ids.append(target['im_id']) # item index in the obj folder

        assert len(paths_rgb)>0, "rgb images not found."

        paths = {
            "rgb" : paths_rgb,
            "depth" : paths_depth,
            "gt_mask" : paths_mask,
            "pred_mask": [] 
            }

        return paths, list_obj_ids, list_im_ids

    def get_paths(self):
        '''
        get data path of LineMOD dataset
        '''
        paths_rgb = []
        paths_depth = []
        paths_mask = []
        paths_pred_mask = []

        list_obj_ids = []
        list_im_ids = [] 

        for index in self.obj_ids:
            if self.select_obj is not None:
                if not index in self.select_obj:
                    continue
            item = '%02d' % index
            cur_dir = os.path.join(self.root_dir, 'data', item)
            
            gt_file = os.path.join(cur_dir, 'gt.yml') # ground truth meta file: pose, bbox
            list_file = open( os.path.join(self.root_dir, 'data', \
                item, self.split+'.txt') )

            line = list_file.readline().strip()
            while line:
                rgb_file = os.path.join(cur_dir, 'rgb', line + '.png')
                depth_file = os.path.join(cur_dir, 'depth', line + '.png')
                mask_file = os.path.join(cur_dir, 'mask', line + '.png') # ground truth segmentation
                
                paths_rgb.append(rgb_file)
                paths_depth.append(depth_file)
                paths_mask.append(mask_file)
                if self.load_seg_pred is not None:
                    # example path: self.load_seg_pred = '../checkpoints/experiment_name/images_test/segnet_results'
                    # path to segnet results: os.path.join(self.root_dir, 'segnet_results')
                    # mask_file = os.path.join('../checkpoints/seg_linemod/images_val', item, line + '.png')
                    # pred_mask_file = os.path.join(self.load_seg_pred, item+'_label', line+'_label.png')
                    pred_mask_file = os.path.join(self.load_seg_pred, item, 'mask', line+'.png')
                    paths_pred_mask.append(pred_mask_file)

                list_obj_ids.append(index) # class index of each item
                list_im_ids.append(int(line)) # item index within corresponding class

                line = list_file.readline().strip()

        assert len(paths_rgb)>0, "rgb images not found."

        
        size = -1#10 
        paths = {
           "rgb" : paths_rgb[:size],
           "depth" : paths_depth[:size],
           "gt_mask" : paths_mask[:size],
           'pred_mask': paths_pred_mask[:size]
           }

        return paths, list_obj_ids, list_im_ids

    def get_meta(self):
        '''
        get meta data: rotation, translation, bbox
        '''
        meta = {}
        for index in self.obj_ids:
            item = '%02d' % index
            cur_dir = os.path.join(self.root_dir, 'data', item)
            meta_file = os.path.join(cur_dir, 'gt.yml')

            if index == 2:
                meta_raw = yaml.load(open(meta_file), Loader=Loader)
                meta_extracted = {}
                for i in range(0, len(meta_raw)):
                    for j in range(len(meta_raw[i])):
                        if meta_raw[i][j]['obj_id'] == 2:
                            meta_extracted[i] = [meta_raw[i][j]]
                            break
                meta[index] = meta_extracted
            else:
                meta[index] = yaml.load(open(meta_file), Loader=Loader)

        return meta

    # we don't really need bbox, we assume there is only one object in each frame
    # the segmentation mask contains all information we need
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

    # def get_mask_inside_bbox(self, mask, bbox_index):
    #     mask_in_bbox = mask[bbox_index].flatten().nonzero()[0]
    #     if len(mask_in_bbox) == 0:
    #         cc = 0
    #         return (cc, cc, cc, cc, cc, cc)

    #     if len(mask_in_bbox) > self.num_points:
    #         c_mask = np.zeros(len(mask_in_bbox), dtype=int)
    #         c_mask[:self.num_points] = 1
    #         np.random.shuffle(c_mask)
    #         mask_in_bbox = mask_in_bbox[c_mask.nonzero()]
    #     else:
    #         pass#mask_in_bbox = np.pad(mask_in_bbox, (0, self.num_points - len(mask_in_bbox)), 'wrap')

    #     return mask_in_bbox

    def get_pose_from_meta(self, index):
        obj = self.list_obj_ids[index]
        im_index = self.list_im_ids[index]
        meta = self.meta[obj][im_index][0]

        trans = np.array(meta['cam_t_m2c']) / self.unit_scale # in meter
        rot = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
        pose = np.concatenate((rot, trans[np.newaxis].T), 1) # 3 x 4 matrix
        return pose

    def __getraw__(self, index):
        r"""data read directly from dataset

        """
        rgb = rgb_read(self.paths['rgb'][index])
        depth = depth_read(self.paths['depth'][index])
        gt_mask_raw = mask_read(self.paths['gt_mask'][index])
        gt_mask = self.remove_invalid_label(gt_mask_raw, depth)
        pose = self.get_pose_from_meta(index)
        
        obj = self.list_obj_ids[index]
        candidates = {
           'rgb': rgb,
           'depth': depth,
           'gt_mask': gt_mask, # ground truth segmentation
           'pose': pose,
           'model': self.models[obj],
        }
        if self.load_seg_pred:
            pred_mask = mask_read(self.paths['pred_mask'][index])
            pred_mask = self.remove_invalid_label(pred_mask, depth)
            candidates['pred_mask'] = pred_mask
        return candidates
