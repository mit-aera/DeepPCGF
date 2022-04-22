# -*- coding: utf-8 -*-
# tf_torch.py
#
# Geometric conversions, some are adapted from PyTorch Geometry
#

import numpy as np
from math import sqrt
import torch
import torch.nn as nn
from numpy import linalg as LA

import utils.eval_utils as eval_utils
import utils.open3d_utils as o3d_utils

__all__ = [
    # functional api
    "_EPS",
    "identity_mat",
    "assemble_pose",
    "convert_points_from_homogeneous",
    "convert_points_to_homogeneous",
    "rotation_matrix_to_quaternion",
]

"""epsilon
"""
_EPS=1e-10

def identity_mat(bs, dim):
    r"""
    args:
        bs: batch_size
        dim: dimension
    """
    return torch.eye(dim).reshape(1, 4, 4).repeat(bs, 1, 1)

def assemble_pose(rot, tvec, device):
    r"""combine rotation matrix and translation vector to homogeneous pose matrix
    args:
        rot: (bs, 3, 3)
        tvec: (bs, 3)
    """
    bs = rot.shape[0]
    pose = identity_mat(bs, 4).to(device)
    pose[:, :3, :3] = rot
    pose[:, :3, 3] = tvec

    return pose

def convert_points_from_homogeneous(points):
    r"""Function that converts points from homogeneous to Euclidean space.

    See :class:`~torchgeometry.ConvertPointsFromHomogeneous` for details.

    Examples::

        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> output = tgm.convert_points_from_homogeneous(input)  # BxNx2
    """
    if not torch.is_tensor(points):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(points)))
    if len(points.shape) < 2:
        raise ValueError("Input must be at least a 2D tensor. Got {}".format(
            points.shape))

    return points[..., :-1] / points[..., -1:]

def convert_points_to_homogeneous(points):
    r"""Function that converts points from Euclidean to homogeneous space.

    See :class:`~torchgeometry.ConvertPointsToHomogeneous` for details.

    Examples::

        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> output = tgm.convert_points_to_homogeneous(input)  # BxNx4
    """
    if not torch.is_tensor(points):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(points)))
    if len(points.shape) < 2:
        raise ValueError("Input must be at least a 2D tensor. Got {}".format(
            points.shape))

    return nn.functional.pad(points, (0, 1), "constant", 1.0)

def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (1 - mask_d0_d1)
    mask_c2 = (1 - mask_d2) * mask_d0_nd1
    mask_c3 = (1 - mask_d2) * (1 - mask_d0_nd1)
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q

############################################################################   

#
# untested functions below
#
def inv_transform_pts(self, rot_mat, t_vec, pts):
    '''Inverse transformation
    '''
    t_vec = t_vec.view(3,1)
    
    assert rot_mat.shape == torch.Size([3,3])
    assert t_vec.shape == torch.Size([3,1])
    assert pts.shape == torch.Size([1, 500,3])
   
    bs, num_pts, _, = pts.shape

    R = torch.transpose(rot_mat, 0, 1)[None, :, :].repeat(num_pts, 1, 1).view(num_pts, 3, 3)
    pts = pts - t_vec.squeeze()
    transformed_pts = torch.matmul(R, pts[:, :, :, None]).squeeze(3)

    return transformed_pts

def transform_pts(quat, tvec, pts):
    assert quat.shape == torch.Size([1,4])
    # print('tvec', tvec.shape)
    # print('pts', pts.shape)
    bs, num_pts, _ = pts.shape
    # print('res', a.shape)
    pts_new = qrot(quat[None, :].expand(-1, num_pts, -1), pts) + tvec.squeeze()[None, None, ].expand(-1, num_pts, -1)
    # new_t = new_t.transpose(0, 1)
    return pts_new

def quaternion2matrix(quaternion, homogeneous=False):
    pred_r = quaternion.squeeze()
   
    #verified
    rot_matrix = torch.tensor([
        [1.0 - 2.0*(pred_r[2]**2 + pred_r[3]**2),\
        2.0*pred_r[1]*pred_r[2] - 2.0*pred_r[0]*pred_r[3], \
        2.0*pred_r[0]*pred_r[2] + 2.0*pred_r[1]*pred_r[3]], \
        [2.0*pred_r[1]*pred_r[2] + 2.0*pred_r[3]*pred_r[0], \
         1.0 - 2.0*(pred_r[1]**2 + pred_r[3]**2), \
        -2.0*pred_r[0]*pred_r[1] + 2.0*pred_r[2]*pred_r[3]], \
        [-2.0*pred_r[0]*pred_r[2] + 2.0*pred_r[1]*pred_r[3], \
        2.0*pred_r[0]*pred_r[1] + 2.0*pred_r[2]*pred_r[3], \
        1.0 - 2.0*(pred_r[1]**2 + pred_r[2]**2)]
            ], requires_grad=True)
    return rot_matrix
    
    q = quaternion.squeeze().clone()
    n = torch.matmul(q, q)
    if n.item() < _EPS:
        return torch.eye(3)
    q *= torch.sqrt(2.0 / n)
    q = torch.ger(q, q)

    if homogeneous:
        rot_matrix = torch.tensor([
            [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0],
            [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0],
            [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0],
            [                  0,                   0,                   0, 1]
           ])
    else:
        rot_matrix = torch.tensor([
            [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
            [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
            [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]],
           ], requires_grad=True)
   
    return rot_matrix

def quaternion_conjugate(quaternion):
    """Return conjugate of quaternion.
    
    args:
        quaternion: tensor (1,4)
    >>> q0 = random_quaternion()
    >>> q1 = quaternion_conjugate(q0)
    >>> q1[0] == q0[0] and all(q1[1:] == -q0[1:])
    True

    """
    q = torch.tensor(quaternion, dtype=torch.float32)
    q[:, 1:] = -q[:, 1:]
    return q

def quaternion_multiply(q1, q0):
    """Return multiplication of two quaternions. R = R1 * R0

    args:
        q0: w0, x0, y0, z0
        q1: w1, x1, y1, z1

    >>> q = quaternion_multiply([4, 1, -2, 3], [8, -5, 6, 7])
    >>> numpy.allclose(q, [28, -44, -14, 48])
    True

    """
    q1 = q1.squeeze()
    q0 = q0.squeeze()
    return torch.tensor([
        -q1[1]*q0[1] - q1[2]*q0[2] - q1[3]*q0[3] + q1[0]*q0[0],
        q1[1]*q0[0] + q1[2]*q0[3] - q1[3]*q0[2] + q1[0]*q0[1],
        -q1[1]*q0[3] + q1[2]*q0[0] + q1[3]*q0[1] + q1[0]*q0[2],
        q1[1]*q0[2] - q1[2]*q0[1] + q1[3]*q0[0] + q1[0]*q0[3]], dtype=torch.float32)

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    
    # print('quat shape', q.shape)
    # print('tvec', v.shape)
    

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)
    
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

def rotate_point_by_quaternion(quat, point):
    r = torch.cat((torch.zeros(1, 1, dtype=point.dtype).cuda(), point), dim=1)
    # print('r', r.shape)
    quat_conj = quaternion_conjugate(quat)
    return quaternion_multiply(quaternion_multiply(quat,r), quat_conj)[1:]

def transform_points(trans_01: torch.Tensor,
                     points_1: torch.Tensor) -> torch.Tensor:
    r"""Function that applies transformations to a set of points.

    Args:
        trans_01 (torch.Tensor): tensor for transformations of shape
          :math:`(B, D+1, D+1)`.
        points_1 (torch.Tensor): tensor of points of shape :math:`(B, N, D)`.
    Returns:
        torch.Tensor: tensor of N-dimensional points.

    Shape:
        - Output: :math:`(B, N, D)`

    Examples:

        >>> points_1 = torch.rand(2, 4, 3)  # BxNx3
        >>> trans_01 = torch.eye(4).view(1, 4, 4)  # Bx4x4
        >>> points_0 = tgm.transform_points(trans_01, points_1)  # BxNx3
    """
    if not torch.is_tensor(trans_01) or not torch.is_tensor(points_1):
        raise TypeError("Input type is not a torch.Tensor")
    if not trans_01.device == points_1.device:
        raise TypeError("Tensor must be in the same device")
    if not trans_01.shape[0] == points_1.shape[0]:
        raise ValueError("Input batch size must be the same for both tensors")
    if not trans_01.shape[-1] == (points_1.shape[-1] + 1):
        raise ValueError("Last input dimensions must differe by one unit")
    # to homogeneous
    points_1_h = convert_points_to_homogeneous(points_1)  # BxNxD+1
    # transform coordinates
    points_0_h = torch.matmul(
        trans_01.unsqueeze(1), points_1_h.unsqueeze(-1))
    points_0_h = torch.squeeze(points_0_h, dim=-1)
    # to euclidean
    points_0 = convert_points_from_homogeneous(points_0_h)  # BxNxD
    return points_0

def combine_two_poses(quat_a, t_a, quat_b, t_b):
    '''
    [quat_b, t_b] x [quat_a, t_a]
    args:
        quat_a, quat_b: bsx4
        t_a, t_b: bsx3
    '''
    new_quat = quaternion_multiply(quat_b, quat_a)
    new_t = rotate_point_by_quaternion(quat_b, t_a).cuda() + t_b.squeeze()
    return new_quat, new_t

# def combine_two_poses(init_quat, init_t, delta_quat, delta_t):
#     print('init_quat', init_quat)
#     print('delta_quat', delta_quat)
#     print('init_t', init_t)
#     print('delta_t', delta_t)
#     init_rot = quaternion2matrix(init_quat, homogeneous=True)
#     delta_rot = quaternion2matrix(delta_quat, homogeneous=True)


#     init_T = init_rot.clone()
#     init_T[:3, 3] = init_t.squeeze()
#     delta_T = delta_rot.clone()
#     delta_T[:3, 3] = delta_t.squeeze()

#     final_T = torch.matmul(delta_T, init_T)
#     print('final_T', final_T)
#     final_rot = final_T[:3, :].clone()
#     final_rot[0, 3] = 0
#     final_rot[1, 3] = 0
#     final_rot[2, 3] = 0
#     new_quat = rotation_matrix_to_quaternion(final_rot[None, :, :])
#     new_t = final_T[:3, 3]
#     # print(new_quat.shape)
#     # print(new_t.shape)
#     return new_quat, new_t

def line_intersection(pts_src, dir_vecs):
    '''Find intersection point of a vector field in 3D space,
    in the least squares sense. 
    Adapted from the system given by eq. 13 in 
    http://cal.cs.illinois.edu/~johannes/research/LS_line_intersect.pdf.
    See also https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#In_three_dimensions_2

    args:
        pts_src: Nx3 array containing starting points of the vector field
        dir_vecs: direction vector of each starting point
    return:
        pt_intersect: best intersection point of the N vectors, 
          in the least squares sense.
    '''

    # generate all line direction vectors 
    # print('norm', np.linalg.norm(dir_vecs,axis=1)[:,np.newaxis]) 
    n = dir_vecs/np.linalg.norm(dir_vecs,axis=1)[:,np.newaxis] # normalized

    # generate the array of all projectors 
    a = n[:,:,np.newaxis]*n[:,np.newaxis]
    
    projs = np.eye(n.shape[1]) - n[:,:,np.newaxis]*n[:,np.newaxis]  # I - n*n.T

    # generate R matrix and q vector
    R = projs.sum(axis=0)
    a = projs @ pts_src[:,:,np.newaxis]
    q = (projs @ pts_src[:,:,np.newaxis]).sum(axis=0)

    # solve the least squares problem for the 
    # intersection point p: Rp = q
    pt_intersect = np.linalg.lstsq(R,q,rcond=None)[0]
    return pt_intersect.T


# Input: expects Nx3 matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform_3D(A, B):
    # print('gt', A.shape)
    # print('pred', B.shape)
    assert len(A) == len(B)
    N = A.shape[0]; # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.matmul( np.transpose(AA), BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T,  U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
       # print "Reflection detected"
       Vt[2,:] *= -1
       R = Vt.T * U.T
    t = -R.dot(centroid_A[None, :].T) + centroid_B[None, :].T

    return R, t

def apply_transform(pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts

def geometric_check(src, tgt, index, inlier_threshold):
    c1 = np.mean(src[index, :], axis=0, keepdims=True)
    c2 = np.mean(tgt[index, :], axis=0, keepdims=True)
    d1 = LA.norm(src - c1, axis=1)
    d2 = LA.norm(tgt - c2, axis=1)
    return abs(d1 - d2) < inlier_threshold

def ransac_find_transform(src, tgt, k=3, max_iteration=10000, inlier_threshold=0.012):
    r"""given point correspondences, find best transform between two point clouds
         with ransac method
    src: model points
    """
    transform = o3d_utils.ransac_global_correspondence(src, tgt)
    best_R = transform[:3, :3]
    best_t = transform[:3, 3:4]
    return best_R, best_t    
    # original implementation
    # num_pts = src.shape[0]
    # best_ic = 0
    # best_R = None
    # best_t = None
    # goal_inliers = 500
    # best_index = None
    # min_error = np.inf
    # for i in range(max_iteration):
    #    index = np.random.choice(num_pts, k)
    #    R, t = rigid_transform_3D(src[index,:], tgt[index, :])
    #    tgt_pred = apply_transform(src, np.concatenate((R, t), axis=1))       
       
    #    error_array = LA.norm(tgt - tgt_pred, axis=1)

    #    ii = eval_utils.compute_min_distance(
    #             R, t, 
    #             tgt, src) < inlier_threshold
    #    inlier_index = error_array < inlier_threshold

    #    #ii = geometric_check(src, tgt, index, inlier_threshold*0.1)
    #    # inlier_index = np.multiply(inlier_index, ii)

    #    #print('inlier', inlier_index.max(), inlier_index.shape)

       


    #    ic = np.count_nonzero(inlier_index) 
    #    #error_sum = np.sum(error_array)
    #    if ic > best_ic:# and distance < min_error:
    #        best_ic = ic
    #        #min_error = distance
    #        best_index = inlier_index 
    #        if ic > goal_inliers:
    #            break
    # best_R, best_t = rigid_transform_3D(src[best_index,:], tgt[best_index, :])
     
    

# Test with random data
if __name__ == '__main__':  
    '''verified pose composition
    '''
    # a = delta_quat.squeeze().cpu().numpy()
    # delta_pose = quaternion_matrix(a)
    # delta_pose[:3, 3] = delta_t.squeeze().cpu().numpy().T
    # init_pose = np.concatenate((init_R.cpu().numpy(), init_t.cpu().numpy()), axis=1)
    # init_pose = np.concatenate((init_pose, np.array([[0, 0, 0, 1]])), axis=0)
    # new_pose = delta_pose @ init_pose
    # print('new_pose', new_pose)
    # print('R', quaternion_matrix(self.final_quat.cpu().numpy()))
    # print('t', self.final_t)
    
    '''verified SVD decomposition
    '''  
    # Random rotation and translation
    R = np.mat(np.random.rand(3,3))
    t = np.mat(np.random.rand(3,1))
    
    # make R a proper rotation matrix, force orthonormal
    U, S, Vt = np.linalg.svd(R)
    R = U*Vt
    
    # remove reflection
    if np.linalg.det(R) < 0:
       Vt[2,:] *= -1
       R = U*Vt
    
    # number of points
    n = 10
    
    A = np.mat(np.random.rand(n,3));
    B = R*A.T + np.tile(t, (1, n))
    B = B.T;
    
    # recover the transformation
    ret_R, ret_t = rigid_transform_3D(A, B)
    
    A2 = (ret_R*A.T) + np.tile(ret_t, (1, n))
    A2 = A2.T
    
    # Find the error
    err = A2 - B
    
    err = np.multiply(err, err)
    err = np.sum(err)
    rmse = np.sqrt(err/n);
    
    print("Points A")
    print(A)
    print("")
    
    print("Points B")
    print(B)
    print("")
    
    print("Rotation")
    print(R)
    print("")
    
    print("Translation")
    print(t)
    print("")
    
    print("RMSE:", rmse)
    print("If RMSE is near zero, the function is correct!")

# def solvePnPRansac(pts_obj, pts_img, camera_matrix):
#     _, rvec, tvec, inliers = cv2.solvePnPRansac(pts_obj, pts_img, camera_matrix, distCoeffs=None)

#     rot, jac = cv2.Rodrigues(rvec)

#     return rot, tvec
