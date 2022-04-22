import argparse
import os
import torch

from utils import logger_utils
import models


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='../checkpoints', help='models are saved here')
        parser.add_argument('--resume', type=str, help='load from specified checkpoint')

        # model parameters
        parser.add_argument('--model', type=str, default='pointreg', help='chooses which model to use. [pointreg | matching]')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--gcn_nfeat', type=int, default=64, help='# of feature channels of GCN')
        parser.add_argument('--gcn_nhid', type=int, default=16, help='# of hidden layers of GCN')
        parser.add_argument('--gcn_dropout', type=float, default=0.5, help='# of feature channels of GCN')


        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--geometric_check', type=str, default='vanilla', help='chooses which geometric check to use. [vanilla | ransac | mc]')
        parser.add_argument('--select_pts', type=int, default='-1', help='choose k best points from gcn')


        # parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        # dataset parameters
        parser.add_argument('--use_gt_seg', action='store_true',
            help='if specified, ground truth segmentation is used')
        parser.add_argument('--dataset', type=str, default='LineMOD', help='option: LineMOD | YCBVideo')
        parser.add_argument('--select_obj', type=str, help='select which objects to load')
        parser.add_argument('--data_path', type=str, default='../data/Linemod_preprocessed', metavar='PATH', help='path to the data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--voxel_size', type=float, default=1e-3, help='voxel quantization size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--workers', '-w', type=int, default=8,
            help='number of cpu threads to use during batch generation (default: 8)')
        parser.add_argument('--image_height', type=int, default=480, help='size of image height')
        parser.add_argument('--image_width', type=int, default=640, help='size of image width')
        parser.add_argument('--num_points', type=int, default=1500, help='number of points sampled from point clouds')
        # parser.add_argument('--hit_ratio_thresh', type=float, default=0.01, help='hit ratio threshold')
        parser.add_argument('--image_based', default=False, help='if specified, do inference on image (e.g. segmentation task)')

        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='-1', help='which iteration to load? if load_iter >= 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

        # visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        # parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        # parser.add_argument('--display_port', type=int, default=6006, help='tensorboard port of the web display')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.initialized = True

        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        # dataset_name = opt.dataset_mode

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        logger_utils.mkdirs(expr_dir)
        if opt.isTrain:
            file_name = os.path.join(expr_dir, 'train_opt.txt')
        else:
            file_name = os.path.join(expr_dir, 'test_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
