from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser, isTrain=False):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        parser.add_argument('--do_icp_refine', action='store_true', 
            help='if specified, pose is refined by icp')
        parser.add_argument('--save_seg', action='store_true', 
            help='if specified, save predicted segmentation mask to disk')

        parser.add_argument('--outlier_ratio', type=float, default=0.0,
            help='outlier ratio')

        self.isTrain = isTrain
        return parser
