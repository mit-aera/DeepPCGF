import numpy as np

class CameraModel:
    r"""(perspective) Camera projection model
    """
    def __init__(self, cam_cx, cam_cy, cam_fx, cam_fy):
        # camera parameters
        self.cam_cx = cam_cx
        self.cam_cy = cam_cy
        self.cam_fx = cam_fx
        self.cam_fy = cam_fy
        self.matrix_K = np.array([[self.cam_fx,            0., self.cam_cx],
                                  [          0., self.cam_fy,  self.cam_cy],
                                  [          0.,           0.,           1.]])
        
    def cam2img(self, pts_c):
        '''project points in camera frame to image
        args:
            pts_c: Nx3 array, coordiantes in camera frame
        return:
            pts_i: Nx2 array, projected pixel coordinates
        '''
        
        pts_i = np.matmul(pts_c, self.matrix_K.T)
        mask = pts_i[:, 2]>0
        pts_i[mask, :2] = pts_i[mask, :2] / pts_i[mask, 2:]
        return pts_i

    def img2cam(self, depth, xmap, ymap):
        r"""inverse function of cam2img
        """
        pt_z = depth
        pt_x = (xmap - self.cam_cx) * pt_z / self.cam_fx
        pt_y = (ymap - self.cam_cy) * pt_z / self.cam_fy
        pts_cam = np.concatenate((pt_x, pt_y, pt_z), axis=1) 
        return pts_cam