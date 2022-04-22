import os
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from models.seg_network import SegNetwork
from models.resunet import CoordRegNet 
from models.gcn_network import GCN


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=opt.lr_decay_gamma)
    elif opt.lr_policy == 'multi_step':
        #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[200,300,400,500], gamma=0.5)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[200, 250, 300, 350, 400, 450], gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=opt.factor, threshold=opt.threshold, patience=opt.patience)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    elif opt.lr_policy == 'exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    else:
        raise NotImplementedError('learning rate policy [%s] is not recognized', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], data_parallel=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if data_parallel:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

def load_networks(checkpoints_dir, experiment_name, model_names, nets, epoch, device):
    """Load all the networks from the disk.
    Parameters:
        epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
    """
    model_dir = os.path.join(checkpoints_dir, experiment_name)
    for i, name in enumerate(model_names):
        if isinstance(name, str):
            load_filename = 'epoch_%s_net_%s.pth' % (epoch, name)
            load_path = os.path.join(model_dir, load_filename)
            if not os.path.exists(load_path):
                print(load_path)
                raise ValueError('checkpoint not found')
            net = nets[i] 
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            print('loading the model from %s' % load_path)
            checkpoint = torch.load(load_path, map_location=str(device))
            try:
                state_dict = checkpoint['net']
            except:
                state_dict = checkpoint
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata

            #net.load_state_dict(state_dict)

            pre_trained = list(state_dict.items())
            cur_model_kvpair = net.state_dict()
            count = 0
            for key, value in cur_model_kvpair.items():
                layer_name, weights = pre_trained[count]
                cur_model_kvpair[key] = weights
                count += 1
            net.load_state_dict(cur_model_kvpair)

def define_coordreg_net(num_classes, init_type='normal', init_gain=0.02, gpu_ids=[0]):
    """Create a coordinate regression network 

    Parameters:
        num_classes: number of output channels
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a CoordRegNet
    """    
    net = CoordRegNet(num_classes, D=3)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_seg_net(nobj, init_type='normal', init_gain=0.02, gpu_ids=[0]):
    """Create a segmentation network 

    Parameters:
        model (str) -- the name of model
        nobj (int) -- the number of object model to train
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a posenet
    """
    # net = PoseNet(num_points = npt, num_obj = nobj)
    # net = SegNet(n_classes=nobj) 
    net = SegNetwork(input_nbr=1, label_nbr=nobj*2)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_gcn_net(nobj, nfeat, nhid, dropout, init_type='normal', init_gain=0.02, gpu_ids=None):
    """Create a graph convolutional network 

    Parameters:
        model (str) -- the name of model
        nobj (int) -- the number of object model to train
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a gcn 
    """
    net = GCN(nfeat=nfeat, nhid=nhid, nclass=nobj, dropout=dropout)
    return init_net(net, init_type, init_gain, gpu_ids, data_parallel=False)
