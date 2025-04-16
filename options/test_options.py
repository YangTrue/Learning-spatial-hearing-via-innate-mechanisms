from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        """
        task:
        1. phase and isTrain 
        2. ntest and num_test 
        """
        parser = BaseOptions.initialize(self, parser)  # define shared options
        
        # set default for test procedure
        parser.set_defaults(phase='test')

        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.') # where did we use this?
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        
        parser.add_argument('--interactive_imagedir', type=str, default='', help='used only in interactive test, to give the image path')
        parser.add_argument('--interactive_imagename', type=str, default='', help='used only in interactive test, to give the image filename')
        parser.add_argument('--interactive_cmd', type=int, default=6, help='used for other interactive test instructions, e.g. the target digits for the blender models')
        
        parser.add_argument('--test_visual', action='store_true', help='save output visual during test')
        
        self.isTrain = False
        return parser
