import numpy as np
from sklearn.neighbors import KDTree
from data.tf_numpy import quaternion_distance, quaternion_angle



def compute_adds_metric(rot_pred, trans_pred, rot_gt, trans_gt, 
      src, is_symmetric):
    '''
    compute the mean of closest point distance
    '''
    rot_gt = rot_gt
    trans_gt = trans_gt
    assert rot_pred.shape == (3,3)
    assert trans_pred.shape == (3,1)
    assert rot_gt.shape == (3,3)
    assert trans_gt.shape == (3,1)

    src_pred = np.dot(src, rot_pred.T) + trans_pred.T
    model_gt = np.dot(src, rot_gt.T) + trans_gt.T
    if is_symmetric:
        src_pred = src_pred.astype(np.float32)
        model_gt = model_gt.astype(np.float32)
        tree = KDTree(model_gt, leaf_size=2)
        dist, ind = tree.query(src_pred, k=1)
        model_gt = np.take(model_gt, np.squeeze(ind), axis=0)
        dis = np.mean(np.linalg.norm(src_pred - model_gt, axis=1)) 
    else:
        dis = np.mean(np.linalg.norm(src_pred - model_gt, axis=1))

    return dis

def is_correct_pred(dis_pred, model_diameter, threshold=0.1):
    '''
    The 6D pose is considered to be correct if the average distance 
        is smaller than a predefined threshold (default is 10%)
    '''
    return dis_pred < model_diameter*threshold

def compute_error(best_quat, best_t, gt_quat, gt_t):
    assert best_t.shape == (3, 1)
    assert gt_t.shape == (3, 1)
    x_offset, y_offset, z_offset = abs(best_t - gt_t)
    q_distance = quaternion_distance(best_quat, gt_quat)
    q_angle = quaternion_angle(best_quat, gt_quat)
    return q_distance, q_angle, x_offset[0], y_offset[0], z_offset[0]

def rte(t_pred, t_gt):
    r"""Relative translation error
    """
    rte = np.linalg.norm(t_pred - t_gt)
    return rte

def rre(r_pred, r_gt, in_degree=False):
    r"""Relative Rotation Error (RRE)
    args:
        - r_pred: predicted rotation matrix
        - r_gt: ground truth rotation matrix
    """
    
    try:
        #rre = np.arccos((np.trace(r_pred.T @ r_gt) - 1) / 2)
        rre = abs(np.arccos(min(max(
            ((np.matmul(r_pred.T, r_gt)).trace() - 1.0) / 2.0, -1.0), 1.0))) 
    except:
        print('cos theta', (np.trace(r_pred.T @ r_gt) - 1) / 2)
    if in_degree:
        rre = rre / np.pi * 180
    return rre

class IoU:
    r"""Computes the intersection over union (IoU) per class and corresponding
    mean (mIoU).

    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:

        IoU = true_positive / (true_positive + false_positive + false_negative).

    Keyword arguments:
    - num_classes (int): number of classes in the classification problem
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes

        # res: stored results, col 0: sum; col 1: # of cases; col 2: average
        self.res = np.zeros((num_classes, 3))

    def _eval_iou(self, pred_mask, gt_mask):
        if np.sum(pred_mask) == 0:
            if np.sum(gt_mask) == 0:
                return 1
            else:
                return 0
        if np.sum(gt_mask) == 0:
            return 0

        n_ii = np.sum(np.logical_and(pred_mask, gt_mask))
        t_i  = np.sum(gt_mask)
        n_ij = np.sum(pred_mask)

        iou = n_ii / (t_i + n_ij - n_ii)
        return iou

    def add(self, pred_label, gt_label, class_id):
        r"""Add one test case
        args:
            pred_label: (H, W)
            gt_label: (H, W)
        """
        iou = self._eval_iou(pred_label, gt_label)

        self.res[class_id, 0] += iou
        self.res[class_id, 1] += 1
        self.res[class_id, 2] = self.res[class_id, 0] / self.res[class_id, 1]
