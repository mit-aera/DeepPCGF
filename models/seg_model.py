import torch
import torch.nn as nn
import torch.nn.parallel

from .base_model import BaseModel
from models.networks import define_seg_net



class SegModel(BaseModel):
    """ This class implements segmentation model 

    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        args:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        returns:
            the modified parser.

        """
        args = parser.parse_args()
        parser.set_defaults(image_based=True)  # this model only support inference on rgb or depth image
        parser.add_argument('--depth', action='store_true', 
                help='if specified, do segmentation on depth image; otherwise on rgb image')
        if not is_train:
            return parser
        if args.lr_policy == 'step':
            parser.add_argument('--lr_decay_iters', type=int, default=80, 
                help='multiply by a gamma every lr_decay_iters iterations')
            parser.add_argument('--lr_decay_gamma', type=float, default=0.5, 
                help='multiply by lr_decay_gamma every lr_decay_iters iterations')
        elif args.lr_policy == 'linear':
            parser.add_argument('--niter', type=int, default=100, 
                help='# of iter at starting learning rate')
            parser.add_argument('--niter_decay', type=int, default=100, 
                help='# of iter to linearly decay learning rate to zero')
        else:
            raise NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
        return parser

    def __init__(self, opt):
        """Initialize the class.

        args:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out.
        self.loss_names = ['pred']
        # specify the images you want to save/display. 
        self.visual_names = []        
        # define networks
        self.model_names = ['Seg']
        self.seg_depth = True if opt.depth else False 
        
        if opt.dataset == 'LineMOD' or \
            opt.dataset == 'LineMODOcclusion':
            nobj = 13
        elif opt.dataset == 'ycb':
            nobj = 21
        elif opt.dataset == 'BlackBird':
            nobj = 1
        else:
            raise NotImplementedError('dataset [%s] is not implemented', opt.dataset)
        
        self.netSeg = define_seg_net(nobj=nobj, 
            init_type='normal', init_gain=0.02, gpu_ids=opt.gpu_ids)
        
        if self.isTrain:
            self.netSeg.train()
            class_weights = torch.tensor([0.1, 0.9]).to(self.device)
            self.seg_criterion = nn.NLLLoss(weight=class_weights)
            self.optimizer = torch.optim.Adam(self.netSeg.parameters(), 
                lr=opt.lr, betas=(opt.beta1, opt.beta2))

            self.optimizers.append(self.optimizer)
        else:
            self.netSeg.eval()

    def set_phase(self, phase):
        self.current_phase = phase
        if phase == 'train':
            self.netSeg.train()
        else:
            self.netSeg.eval()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        args:
            input (dict): include the data itself and its metadata information.
                - rgb: (bs, 3, H, W)
                - seg_mask: (bs, H, W)
                - gt_cls: (bs, num_class)
        """
        if self.seg_depth:
            self.x = input['depth'].to(self.device)
        else:
            self.x = input['rgb'].to(self.device)
        self.seg_gt = input['gt_mask'].to(self.device)
        obj = input['model'][0] # currently val only supports batch size = 1 
        self.model_index = obj.get_index()


    def forward(self):
        r"""
        rgb: (B, 3, H, W)
        depth: (B, 1, H, W)
        """
        self.netSeg.train()
        out = self.netSeg(self.x) 
        self.seg_log_pred = out[:, self.model_index, :, :, :]
        _, self.seg_out = self.seg_log_pred.max(1)
        
        return self.seg_out
 
    def backward(self):
        seg_loss = self.seg_criterion(self.seg_log_pred, self.seg_gt.detach())
        self.loss_pred = seg_loss 
        self.loss_pred.backward()

    def optimize_parameters(self, step_optimizer):
        self.forward()                  
        self.backward()
        if step_optimizer:
            self.optimizer.step()     
            self.optimizer.zero_grad()
