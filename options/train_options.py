from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser, isTrain=True):
        parser = BaseOptions.initialize(self, parser)
        
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--valid_freq', type=int, default=5, help='frequency of checking the performance')

        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--reset_learning', action='store_true', help='continue training but only load networks')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        parser.add_argument('--do_augmentation', action='store_true', help='whether apply do_augmentation during training')
        # training parameters
        parser.add_argument('--epoch_start', type=int, default=0, help='epoch to start training')
        parser.add_argument('--epoch_end', type=int, default=500, help='epoch to end training')
        #parser.add_argument('--batch_size', '-b', type=int, default=4, help='batch size (default: 4)')
        # parser.add_argument('--val_skip_frames', type=int, default=4, help='skip frames in validation dataset (default: 4)')
        # parser.add_argument('--crop', type=str, default="random", help='bottom | random')        
        parser.add_argument('--sample_interval', type=int, default=50, help='interval between sampling images (default: 50)')
        parser.set_defaults(load_gt=True)

        # optimizer
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--beta1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
        parser.add_argument('--beta2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
        parser.add_argument('--lr_policy', type=str, default='multi_step', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--step_freq', type=int, default='8', help='frequency at which optimizer steps')
        parser.add_argument('--mu', type=float, default=100, help='weight of coordinate loss')

        
        self.isTrain = isTrain
        return parser
