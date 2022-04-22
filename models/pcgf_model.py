import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import MinkowskiEngine as ME

from .base_model import BaseModel
from . import networks
from utils.metrics import AverageMeter
from utils.generate_graph import create_adjacency_mat, create_labels 
from utils.registration import registration
from data.data_utils import apply_transform
from models.networks import load_networks
from models.networks import define_coordreg_net, define_seg_net, define_gcn_net



class PCGFModel(BaseModel):
    """ This class implements the complete pipeline of Pairwise Compatible Geometric Feature model, 
          for estimating pose of input 2.5D point cloud.
          The network predicts 3D coordinates in canonical view
          of each point 

    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        return parser

    def __init__(self, opt):
        """Initialize the class.

        args:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out.
        self.loss_names = ['coord', 'outlier']
        # specify the images you want to save/display. 
        self.visual_names = []        
        # define networks
        self.model_names = ['PR', 'GCN']

        self.geometric_check = opt.geometric_check
        self.device = torch.device('cuda', opt.gpu_ids[0])

        self.npt = opt.num_points
        self.num_classes = 13
        self.use_gt_seg = opt.use_gt_seg
        self.select_pts = opt.select_pts
        self.voxel_size = opt.voxel_size
        self.inlier_threshold = 0.08 
        self.is_weighted = False
        self.alpha = 0
        
        if opt.dataset == 'BlackBird':
            self.num_classes = 1
        self.netSeg = define_seg_net(nobj=self.num_classes,
            init_type=opt.init_type, 
            init_gain=0.02,
            gpu_ids=opt.gpu_ids)
        self.netPR = define_coordreg_net(num_classes=self.num_classes, 
            init_type=opt.init_type, 
            init_gain=0.02, 
            gpu_ids=opt.gpu_ids)
        model_names = ['Seg']
        nets = [self.netSeg]
        load_networks(opt.checkpoints_dir, opt.name, model_names, nets, 0, self.device)
        
        self.netGCN = define_gcn_net(nobj=self.num_classes, 
            nfeat=opt.gcn_nfeat,
            nhid=opt.gcn_nhid,
            dropout=opt.gcn_dropout,
            init_type=opt.init_type, 
            init_gain=0.02, 
            gpu_ids=opt.gpu_ids)
                        
        if self.isTrain:
            self.mu = 1.0
            self.netGCN.train()
            self.netPR.train()

            self.criterion = nn.L1Loss()
            params = list(self.netPR.parameters()) + list(self.netGCN.parameters())
            self.optimizerGCN = torch.optim.Adam(params,
                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizerGCN)
            self.schedulerGCN = networks.get_scheduler(self.optimizerGCN, opt)
            self.schedulers.append(self.schedulerGCN) 

            self.optimizerPR = torch.optim.Adam(params,
                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizerPR)
            self.schedulerPR = networks.get_scheduler(self.optimizerPR, opt)
            self.schedulers.append(self.schedulerPR) 
            
            self.current_phase = 'train'
        else:
            self.current_phase = 'val'
            
        self.forward_timer = AverageMeter('forward_time')
        self.seg_timer = AverageMeter('seg_time')
        self.gc_timer = AverageMeter('pruning_time')
        self.multi_obj = False 

    def set_phase(self, phase):
        self.current_phase = phase
        if phase == 'train':
            self.netGCN.train()
            self.netPR.train()
        else:
            self.netGCN.eval()
            self.netPR.eval()

    def set_dataset(self, dataset):
        self.dataset = dataset 

    def get_model_pts(self, list_models):
        num_pts = list_models[0].get_model_points().shape[0]
        model_points = torch.zeros(len(list_models), num_pts, 3)
        model_index = []
        for i, obj in enumerate(list_models):
            pts_model = obj.get_model_points()
            model_points[i, :] = torch.from_numpy(pts_model[None, :])
            model_index.append(obj.get_index())

        return model_points, model_index

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.input = input
        
        self.depth = input['depth'].to(self.device)
        self.gt_seg = input['gt_mask'].to(self.device)
        self.model_points, self.model_index = self.get_model_pts(self.input['model'])
        self.model_points = self.model_points.to(self.device) #(bs, N, 3)
        if self.current_phase == 'train': 
            self.gt_pose = input['T_gt']

    @torch.no_grad()
    def get_mask(self):
        start = time.time()
        if self.use_gt_seg:
            pred_mask = self.gt_seg
        else:
            self.netSeg.train()
            seg_out = self.netSeg(self.depth) 
            _, pred_masks = seg_out.max(2) 
            pred_mask = pred_masks[:, self.model_index, :, :]
        if (np.count_nonzero(pred_mask.cpu().numpy()) == 0):
            pred_mask = torch.ones_like(pred_mask)
        end = time.time()
        seg_time = end - start
        self.seg_timer.update(seg_time) 
        return pred_mask

    def get_sparse_input(self, pred_mask):
        pred_mask = pred_mask[0, :].cpu().numpy()
        xyz_s = self.dataset.get_segmented_pts(self.depth.cpu().numpy(), pred_mask)
        xyz_s, coords_s, feats_s, sel_s = self.dataset.voxelization(xyz_s)
        N = feats_s.shape[0]
        feats_s = torch.from_numpy(feats_s).float()
        coords_s = torch.cat((torch.from_numpy(coords_s).int(), torch.ones(N, 1).int()*0), 1)
        sinput_s = ME.SparseTensor(feats_s, coords=coords_s.int()).to(self.device)
        return sinput_s, xyz_s

    def regress_coords(self, pred_mask):
        start = time.time()
        sinput_s, xyz_s = self.get_sparse_input(pred_mask)
        pred_coords, feat = self.netPR(sinput_s, 
            torch.LongTensor(self.model_index).to(self.device) ) 
        end = time.time()
        self.forward_timer.update(end-start)
        return pred_coords, feat, xyz_s
    
    def get_coord_labels(self, xyz_s):
        gt_pose = self.gt_pose[0, :]
        gt_pose = np.concatenate((gt_pose, np.array([[0, 0, 0, 1]])), axis=0)
        inv_gt_pose = np.linalg.inv(gt_pose)
        self.gt_coords = apply_transform(xyz_s, inv_gt_pose)
        self.gt_coords = torch.from_numpy(self.gt_coords).to(self.device).float()
        labels = create_labels(self.pred_coords, self.gt_coords, threshold=2.5*self.voxel_size)
        return labels

    def get_gcn_pred(self):
        start = time.time()
        self.adj_mat = create_adjacency_mat(self.src_coords, 
            self.pred_coords.detach().cpu().numpy(), self.inlier_threshold, self.is_weighted)
        self.cls = self.netGCN(self.feat.F, self.adj_mat.to(self.device), 
            torch.LongTensor(self.model_index))
        self.gc_timer.update(time.time() - start)

    def forward(self): 
        pred_mask = self.get_mask()

        self.pred_coords, self.feat, self.src_coords = self.regress_coords(pred_mask) 
        
        if self.geometric_check == 'gcn':
            self.get_gcn_pred() 
            
        if self.current_phase != 'train':
            pred_pose = registration(self.src_coords, self.pred_coords.cpu().numpy(), 
                self.cls, self.select_pts, self.geometric_check)
            return pred_pose

    def backward(self):
        self.labels = self.get_coord_labels(self.src_coords)
        self.loss_coord = self.criterion(self.pred_coords, self.gt_coords)
        self.loss_outlier = F.nll_loss(self.cls, self.labels.to(self.device))
        self.loss_pred = self.mu * self.loss_coord + self.alpha*self.loss_outlier 
        self.loss_pred.backward()

    def optimize_parameters(self, step_optimizer):
        self.forward()                  
        self.backward()
        if step_optimizer:
            self.optimizerGCN.step()     
            self.optimizerGCN.zero_grad()
