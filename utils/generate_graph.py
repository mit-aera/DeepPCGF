import torch
import numpy as np
from scipy.spatial.distance import pdist
import scipy.sparse as sp


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparse_mx = sparse_mx.astype(np.float32)

    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def find_nn(src, tgt):
    r"""find nearest neighbor of src from tgt

    args:
      src: (N, 3)
      tgt: (N, 3)
    """
    knn = KNearestNeighbor(1)

    src = torch.from_numpy(src.astype(np.float32)).cuda().transpose(1,0).contiguous() # (3, N)
    tgt = torch.from_numpy(tgt.astype(np.float32)).cuda().transpose(1,0).contiguous() # (3, N)

    inds = knn.apply(1, tgt.unsqueeze(0), src.unsqueeze(0))
    inds -= 1

    nn_neighbor = torch.index_select(tgt, 1, inds.view(-1)) #(3, N)
    nn_neighbor = nn_neighbor.transpose(1, 0).contiguous()

    del knn
    
    return nn_neighbor.cpu().numpy(), inds.view(-1) # N 


def affinity_function(p1, p2, type='euclidean', mu=0.25):
    r"""function that encodes edge-to-edge affinity
    args:
        p1, p2: (N, 3), 3D coordinates of nodes 
        N: number of points 
        type: 
            euclidean: 2 norm
            ssq: sum of squre
            gm: Geman-McChure, mu: hyperparameter 
    """
    epsilon = 1e-9
    if type == 'euclidean':
        d1 = pdist(p1, 'euclidean')
        d2 = pdist(p2, 'euclidean')
    elif type == 'ssq':
        d1 = pdist(p1, 'sqeuclidean')
        d2 = pdist(p2, 'sqeuclidean')
    elif type == 'gm':
        pass # TODO
    else:
        raise ValueError('affinity function type is not defined!')

    dist = abs(d1-d2)
    ratio = dist/(d2+epsilon) 

    return dist, ratio
    

def association_graph_edgelist(pts_s, pts_p, threshold=0.08, is_weighted=False):
    r"""generate an association graph based on the distance between each pair
    
    args:
       pts_s: (N, 3), sensor points
       pts_p: (N, 3), predicted coordinate of each sensor points 
    """
    num_pts = pts_s.shape[0]
    #dist, ratio = affinity_function(pts_s, pts_p, type='ssq')
    dist, ratio = affinity_function(pts_s, pts_p, type='euclidean')
    exist_edge = ratio < threshold  
    num_edge = np.count_nonzero(exist_edge)
    index = np.triu_indices(num_pts, k=1)

    K = 100000
    if num_edge > K:
        num_edge = K
        ind = np.argpartition( ratio, K)
        exist_edge = ind[:K]
        num_edge = np.count_nonzero(exist_edge)
    try:
        ei = index[1][exist_edge]
        ej = index[0][exist_edge]
        if is_weighted:
            sigma = 10
            edge_weight = np.exp(-ratio*sigma)
            edge_weight = edge_weight[exist_edge]
            return ei, ej, num_pts, num_edge, edge_weight
        return ei, ej, num_pts, num_edge, np.ones(ei.shape[0])
    except:
        print(num_pts, pts_m_nn.shape, num_edge, index[1].shape, exist_edge.shape)
    
def normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj += sp.coo_matrix(np.eye(adj.shape[0]))
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return (d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)).tocoo()
   
def create_adj_from_edgelist(ei, ej, num_pts, edge_weight):
    adj = sp.coo_matrix( (edge_weight, (ei, ej)), shape=(num_pts, num_pts), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalized_adjacency(adj) 
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

def create_adjacency_mat(pts_s, pts_p, threshold=0.08, weighted=False):
    '''
    args: pts_s: scene points
          pts_p: predicted model points
          threshold
    '''
    ei, ej, num_pts, num_edge, edge_weight = \
      association_graph_edgelist(pts_s, pts_p, threshold, weighted)
    
    adj = create_adj_from_edgelist(ei, ej, num_pts, edge_weight)
    #features = sp.csr_matrix(pts_p, dtype=np.float32)
    #features = sp.csr_matrix(feat, dtype=np.float32)
    #features = normalize(features)
    #features = torch.FloatTensor(np.array(features.todense()))
    #features = feat
    return adj

def create_labels(pts_pred, pts_gt, threshold=0.08):
    #print(pts_pred,shape, pts_gt.shape)
    sum_of_square = torch.sum( (pts_pred - pts_gt)**2, dim=1)
    labels = sum_of_square < threshold**2
    #_, ratio = affinity_function(pts_pred, pts_gt, type='euclidean')
    #labels = ratio < threshold  
    labels = labels.long()
    return labels

def get_weighted_graph(pts_s, pts_p, feat, gt_pts_p=None):
    '''
    args: pts_s: scene points
          pts_p: predicted model points
          #gt_pts_p: ground truth model points
          feat: feature on each point
    '''
    ei, ej, num_pts, num_edge = generate_graph(pts_s, pts_p)
    #print('num_edge', num_edge)
    adj = create_adj_from_edgelist(ei, ej, num_pts)

    #features = sp.csr_matrix(pts_p, dtype=np.float32)
    #features = sp.csr_matrix(feat, dtype=np.float32)
    #features = normalize(features)
    #features = torch.FloatTensor(np.array(features.todense()))
    features = feat
   
    if gt_pts_p is not None:
        print(pts_p, gt_pts_p)
        diff = np.linalg.norm(pts_p - gt_pts_p, axis=1) 
        labels = torch.LongTensor(diff < 0.006) 
        return adj, features, labels 
    else:
        return adj, features



def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

if __name__ == '__main__':
    import time

    f1 = np.random.random((100, 3))
    f2 = np.random.random((100, 3))
    f3 = f1 + np.random.random((1, 3))

    adj_mat = create_adjacency_mat(f1, f3, threshold=0.08, weighted=True)

    start = time.time()
    for i in range(100):
        affinity_function(f1, f2, type='euclidean')
    print('time cost of euclidean affinity function', (time.time() - start) / 100.0)

    start = time.time()
    for i in range(100):
        affinity_function(f1, f2, type='ssq')
    print('time cost of ssq affinity function', (time.time() - start) / 100.0)




