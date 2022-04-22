import numpy as np
import torch

from data.tf_torch import rigid_transform_3D, ransac_find_transform
from utils.generate_graph import association_graph_edgelist
#from pmc.pmc import pmc

def registration(src, dst, cls=None, select_pts=70, pruning=None):
    if pruning is None:
        pred_rot, pred_t = rigid_transform_3D(dst, src)
    elif pruning == 'ransac':
        pred_rot, pred_t = ransac_find_transform(dst, src)
    elif pruning == 'mcs': # pmc solver
        ei, ej, num_pts, num_edge, _ = association_graph_edgelist(src, 
            dst, threshold=0.08)
        if num_edge < 6:
            pred_rot, pred_t = rigid_transform_3D(dst, src)
        else:
            index = pmc(ei, ej, num_pts, num_edge)
            pred_rot, pred_t = rigid_transform_3D(dst[index, :],
                src[index, :])
    elif pruning == 'gcn':
        if select_pts == -1:
            index = cls.max(1)[1].cpu().numpy()
            index = np.array(index, dtype=bool)
        else:
            index = torch.topk(cls[:, 1], k=min(select_pts, 
                cls[:, 1].shape[0]), largest=True).indices
            index = index.cpu().numpy()
        if np.count_nonzero(index) > 3:
            pred_rot, pred_t = rigid_transform_3D(dst[index, :],
                src[index, :])
        else:
            pred_rot, pred_t = rigid_transform_3D(dst,
                src)

    else:
        raise ValueError('undefined geometric check method')

    pred_T = np.concatenate((pred_rot, pred_t), axis=1)
    pred_T = np.concatenate((pred_T, np.array([[0, 0, 0, 1]])), axis=0)
    return pred_T