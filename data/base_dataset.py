import numpy as np
from abc import abstractmethod
import torch
import torch.utils.data as data
import MinkowskiEngine as ME
from MinkowskiEngine.utils import sparse_quantize

from data.transforms import image_transforms, \
    pcd_transforms
import utils.open3d_utils as o3d_utils



class BaseDataloader(data.Dataset):
    """
    A base data loader
    """
    def __init__(self, split, root_dir, img_height=480, img_width=640,
        do_augmentation=True, image_based=False):
        '''
        args:
            root_dir: root directory of dataset
            split: 'train' | 'test'

        '''
        self.split = split
        self.root_dir = root_dir
        self.symmetric_obj_idx = []

        # augmentation
        self.do_augmentation = do_augmentation
        self.image_based = image_based

        self.img_transforms = image_transforms(
            mode=split, 
            do_augmentation=do_augmentation)
        self.pcd_transforms = pcd_transforms(
            mode=split, 
            do_augmentation=do_augmentation,
            rotation_range=360)

        self.img_height = img_height
        self.img_width = img_width
        self.ymap = np.array([[j for i in range(self.img_width)] \
            for j in range(self.img_height)])
        self.xmap = np.array([[i for i in range(self.img_width)] \
            for j in range(self.img_height)])

    def get_models(self):
        '''
        collect object model informaiton
        '''
        raise NotImplementedError()

    def get_num_of_models(self):
        return len(self.obj_ids)

    def get_name_of_models(self, model_index):
        return self.obj_dics[self.obj_ids[model_index]]

    # deprecated: confusing
    def is_symmetric_obj(self, model_index):
        model_id = self.obj_ids[model_index]
        return model_id in self.symmetric_obj_id

    # def is_symmetric(self, model_id):
    #     return model_id in self.symmetric_obj_id

    def model2cam(self, pts_m, pose):
        '''tranform points to camera coordinate system
        args:
            pts_m: Nx3 point array, coordiantes in model frame
            pose: 3x4 transformation matrix, from model to cam
        '''
        return np.matmul(pts_m, pose[:3, :3].T) + pose[:3, 3:].T

    def transform_pts(self, pts_m, pose):
        '''tranform points to camera coordinate system
        args:
            pts_m: Nx3 point array, coordiantes in model frame
            pose: 3x4 transformation matrix, from model to cam
        '''
        return np.matmul(pts_m, pose[:3, :3].T) + pose[:3, 3:].T

    def cam2img(self, pts_c):
        '''project points in camera frame to image
        args:
            pts_c: Nx3 array, coordiantes in camera frame
        return:
            pts_i: Nx2 array, projected pixel coordinates
        '''
        return self.cam_model.cam2img(pts_c)

    def model2img(self, pts_m, pose):
        return self.cam2img(self.model2cam(pts_m, pose))

    def get_full_point_cloud(self, depth):
        '''
        recover point cloud of object from RGB image and depth map
        args:
            index: index of object in the image

        '''
        depth = depth.flatten()[:, np.newaxis].astype(np.float32)
        xmap = self.xmap.flatten()[:, np.newaxis].astype(np.float32)
        ymap = self.ymap.flatten()[:, np.newaxis].astype(np.float32)

        pts_sensor_cam = self.cam_model.img2cam(depth, xmap, ymap)
        return pts_sensor_cam

    def get_pts_sensor_cam(self, depth, mask):
        '''
        recover point cloud of object from RGB image and depth map
        args:
            index: index of object in the image

        '''
        depth_masked = depth.flatten()[mask][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap.flatten()[mask][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap.flatten()[mask][:, np.newaxis].astype(np.float32)

        pts_sensor_cam = self.cam_model.img2cam(depth_masked, xmap_masked, ymap_masked) 
        return pts_sensor_cam

    def remove_invalid_label(self, mask, depth):
        r"""Remove pixel where depth<0
        if no valid points found, assume mask of every pixel equal to 1
        mask: segmentation mask
        """
        mask_depth = np.ma.getmaskarray(np.ma.masked_not_equal(depth, 0))
        res = mask * mask_depth
        if np.count_nonzero(res) == 0:
            res = np.ones_like(res)
            print('no object found in mask')
        return res
        
    def get_nonzero_index(self, mask):
        """Filter out nonzero index in the segmentation mask
        """
        nonzero_mask = mask.flatten().nonzero()[0]
        # print('mask', len(nonzero_mask))
        if len(nonzero_mask) == 0:
            raise ValueError('no objects found in segmentation results')

        if len(nonzero_mask) > self.num_points:
            c_mask = np.zeros(len(nonzero_mask), dtype=int)
            c_mask[:self.num_points] = 1
            np.random.shuffle(c_mask)
            nonzero_mask = nonzero_mask[c_mask.nonzero()]
        else:
            pass#mask_in_bbox = np.pad(mask_in_bbox, (0, self.num_points - len(mask_in_bbox)), 'wrap')

        return nonzero_mask

    def get_segmented_pts(self, depth_image, mask):
        nonzero_index = self.get_nonzero_index(mask)
        #!!!!!!!!!! check and remove redundancy
        # print(np.max(mask), np.min(mask), np.count_nonzero(mask), mask_in_bbox.shape)
        # print(nonzero_index == mask_in_bbox)
        pts_sensor_cam = self.get_pts_sensor_cam(depth_image, nonzero_index)
        return pts_sensor_cam

    def get_gt_vector_field(self, pts_farthest, pts_obs):
        '''for each points of observed point cloud, find the direction 
             vector from the point to farthest points
             works for both 3D and 2D

        args: 
            pts_farthest_c: mx3 (or mx2) array
            pts_obs_c: nx3 (or nx2) array
        '''
        # print('input farthest', pts_farthest.shape)
        # print('input obs', pts_obs.shape)
        num_k = pts_farthest.shape[0]
        num_pts = pts_obs.shape[0]
        pts_obs = np.repeat(pts_obs[:, None, :], num_k, axis=1)
        pts_farthest = np.repeat(pts_farthest[None, :, :], num_pts, axis=0)
        
        # print('processed farthest', pts_farthest.shape)
        # print('processed obs', pts_obs.shape)
        dir_vecs = pts_farthest - pts_obs

        norm = np.linalg.norm(dir_vecs, axis=2, keepdims=True)
        norm[norm<1e-3] += 1e-3
        dir_vecs /= norm

        return dir_vecs

    def voxelization(self, xyz):
        r"""processing specific to voxel grid
        """
        # move to the mean location
        center = np.mean(xyz, axis=0)
        xyz -= center

        # Voxelization
        sel = sparse_quantize(xyz / self.voxel_size, return_index=True)
        xyz = xyz[sel]
        
        coords = np.floor(xyz / self.voxel_size)
        coords, indices = np.unique(coords, axis=0, return_index=True)
        
        xyz = xyz[indices]
        
        # Get features
        npts = len(xyz)
   
        feats_train = []
        feats_train.append(np.ones((npts, 1)))
        feats = np.hstack(feats_train)

        xyz += center

        return xyz, coords, feats, sel#, indices
 
    def get_sparse_tensor(self, xyz):
        xyz_batch = []
        coord_batch = []
        feat_batch = []
        for i in range(len(xyz)):
            xyz_s, coords_s, feats_s, sel_s = self.voxelization(xyz[i])
            N = feats_s.shape[0]
            feats_s = torch.from_numpy(feats_s).float()
            coords_s = torch.cat((torch.from_numpy(coords_s).int(), torch.ones(N, 1).int()*i), 1)

            xyz_batch.append(xyz_s)
            coord_batch.append(coords_s)
            feat_batch.append(feats_s)
        #xyz_batch1 = torch.cat(xyz_batch1, 0).float()
        coord_batch = torch.cat(coord_batch, 0).int()
        feat_batch = torch.cat(feat_batch, 0).float()
        sinput_s = ME.SparseTensor(feat_batch, coords=coord_batch)
        return xyz_batch, sinput_s

    def inverse_pose(self, pose):
        R = pose[:3, :3]
        t = pose[:3, 3:4]
        inv_pose = np.eye(4)
        inv_pose[:3, :3] = R.transpose()
        inv_pose[:3, 3:4] = -np.matmul(R.transpose(), t)
        np.testing.assert_allclose (pose @ inv_pose, np.eye(4), atol=1e-6)
        return inv_pose

    def EDN2NED(self, pts):
        tf = np.array([[0., 0., 1., 0.],
                      [1., 0., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 0., 1.]])

        return np.matmul(pts, tf[:3, :3].T) + tf[:3, 3:].T

    def check_points(self, pts_m, pts_s, gt_pose):
        r"""visualize sensor points and tranformed model points
             with gt pose
        """
        pts_m_NED = self.transform_pts(pts_m.copy()+np.array([-0.0, 0.0, 0]), gt_pose)
        # o3d_utils.vis_open3d_two_pcds(pts_m_NED, pts_s+np.array([-0.00, -0.05, 0.00]))
        o3d_utils.vis_open3d_two_pcds(pts_m_NED, pts_s)

    def preprocess_image_based(self, index, raws):
        r"""extra processing after calling __getraw__
            used for segmentation task
        """
        # if torch.rand(1) > 0.5:
        #     raws['rgb'], raws['depth'], raws['gt_mask'] = \
        #         np.fliplr(raws['rgb']), np.fliplr(raws['depth']), np.fliplr(raws['gt_mask'])
        # raws = self.img_transforms(raws)
        # print('3',raws['gt_mask'].max(), raws['gt_mask'].min())
        return (
            raws['rgb'],
            raws['depth'],
            raws['gt_mask'],
            raws['model'],
            raws['pose'],
        )

    def preprocess_pcd_based(self, index, raws):
        '''extra processing after calling __getraw__
            used for point cloud based pose estimation
        # # '''
        # raws = self.img_transforms(raws)
        if 'pred_mask' in raws:
            mask = raws['pred_mask']
        else:
            mask = raws['gt_mask']
        # print(mask.shape)
        # data_utils.show_mask_image(mask)
        #data_utils.show_mask_image(raws['rgb'])
        #data_utils.show_depth_image(raws['depth'] / 50.0*255*20)
        #print(raws['depth'].max(), raws['depth'].min())
        #time.sleep(10)
        
        pts_sensor_cam = self.get_segmented_pts(raws['depth'].copy(), mask)
        
        model_points = raws['model'].get_model_points()
        model_index = raws['model'].get_index()
        pose_gt = np.concatenate((raws['pose'], np.array([[0,0,0,1]]) ), 0)
        
        # pts_full = self.get_full_point_cloud(raws['depth'])
        # if self.dataset == 'BlackBird':
        #     pts_sensor_cam = self.EDN2NED(pts_sensor_cam)
        # self.check_points(model_points, pts_sensor_cam, pose_gt)

        sample = {'xyz': pts_sensor_cam, 'pose_gt': pose_gt}
        sample = self.pcd_transforms(sample)
        pts_sensor_cam = sample['xyz']
        pose_gt = sample['pose_gt']

        xyz_m = model_points.copy()#dutils.sample_knn(model_points, int(self.num_points*0.6), ref_pt='other') #model_points
        xyz_s = pts_sensor_cam.copy()#raws['model']._downsample_points(1000)#model_points 
        # xyz_d = pts_full
       
        # pts_model_cam = self.model2cam(model_points, pose_gt)
        #pts_farthest = raws['model'].get_farthest_points(num_pt=self.num_farthest_pts)
        #pts_farthest_cam = self.model2cam(pts_farthest, pose_gt)
        # gt_dir_s = self.get_gt_vector_field(pts_farthest_cam, xyz_s)

        # pts_farthest_i = self.model2img(pts_farthest, raws['pose'])
        # rgb_selected = rgb_masked.flatten()[mask_in_bbox][:, np.newaxis].astype(np.float32)
        # gt_dir_vecs_i = self.get_gt_vector_field(pts_farthest_i, rgb_selected)
        xyz_m, coords_m, feats_m, sel_m = self.voxelization(xyz_m)
        xyz_s, coords_s, feats_s, sel_s = self.voxelization(xyz_s)
        # xyz_d, coords_d, feats_d, sel_d, indices_d = self.voxelization(xyz_d)
        # Get ground truth directions
        # gt_dir_s = gt_dir_s[sel_s][indices_s]
        # print(xyz_s.shape)
        return (xyz_m, coords_m, feats_m, # CAD model 
                xyz_s, coords_s, feats_s, # obj in sensor frame
                # seg_mask, 
                #pts_farthest, pts_farthest_cam, #gt_dir_s, 
                pose_gt,
                raws['model'],
                raws['rgb'],
                raws['depth'])

    def __len__(self):
        return len(self.paths['rgb']) 

    @abstractmethod
    def __getraw__(self, index):
        raise NotImplementedError()

    def __getitem__(self, index):
        raws = self.__getraw__(index)

        if self.image_based:
            return self.preprocess_image_based(index, raws)
        else:
            return self.preprocess_pcd_based(index, raws)
        # candidates = self.transforms(preprocessed_data)
        # items = {key:val for key, val in candidates.items() if val is not None}
