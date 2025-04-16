import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks

from torchvision.utils import save_image as torch_save_images

class BaseModel(ABC):
    """
    TODO: CUH 20230629
    1. remove the unused code, clean the repo
    
    Model:
    1. static configurations of model: structure, location
    2. operations on parameters: define, init(device), save, load, print and visualisation
    
    Share the Option with other runners. Make modifications if necessary.

    in current version, we have redundant code.
    -----

    This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt, tag='model'):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.isInteractive = opt.isInteractive
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'
        
        # used for saving the best validation
        self.current_val_results = []
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        
        return parser

    # remove these abstract methods
    # @abstractmethod
    # def set_input(self, input):
    #     """Unpack input data from the dataloader and perform necessary pre-processing steps.

    #     Parameters:
    #         input (dict): includes the data itself and its metadata information.
    #     """
    #     pass

    # @abstractmethod
    # def forward(self):
    #     """Run forward pass; called by both functions <optimize_parameters> and <test>."""
    #     pass

    # @abstractmethod
    # def optimize_parameters(self, total_iters):
    #     """Calculate losses, gradients, and update network weights; called in every training iteration.
    #     total_iters are used by potential training switches
    #     """
    #     pass

    def setup(self, opt):
        """obsolete: Load and print networks; create schedulers

        CHU:
        flag isTrain is set by TestOption, which is not used anymore
        both schedulers and loading are refactored

        only printing is used

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    # TODO: where do they use this?
    def eval(self):
        """
        Make models eval mode during test time
        """
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()
    
    def validate(self):
        """ 
        validate the model by
        1. run forward()
        2. calculate loss or accuracy results
        """
        raise NotImplementedError
    
    def init_validate(self):
        """
        init the loss/acc accumulators before each validation epoch
        """
        raise NotImplementedError

    def compare_current_val(self):
        """
        keep a dictionary of validation result, 
        and implement customised comparison between validation,
        return True to save the current network as the best validation model.
        """
        raise NotImplementedError

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def new_save_visuals(self, epoch, nrow, outputdir=None):
        """
        instead of saving to html, we use new utils to save to image grid

        epoch is a string, we can use 'val' for validation
        """
        # TODO how do we save the labels as well?
        # how do we do free drawing?
        if not outputdir:
            outputdir = self.save_dir

        for name in self.visual_names:
            save_filename = '%s_%s.png' % (epoch, name)
            if isinstance(name, str):
                x = getattr(self, name)
                torch_save_images(x.data, os.path.join(outputdir,save_filename), normalize=True, nrow=nrow)
        return

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def save_jit_networks(self, epoch):
        """
        avoid the need of class
        TODO: add a method to load jit to the model? 
        if we have the model class, then we can do better debug?
        that is why we need the validator to have access to the model?
        but sometimes, we do not need that 

        Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_jitnet_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)
                # net is a dataparallel object. need to get the module instead
                model=torch.nn.Sequential(*net.module.layers) # TODO if we have better module definition, no need to sequential layers here
                model_scripted = torch.jit.script(model) # Export to TorchScript. 
                model_scripted.save(save_path) # Save

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)

        Intention:
            Used for special loading
            For more flexible loading, we implement in the new base-model
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def eval(self):
        """
        eval each module in the model
        """
        for n in self.model_names:
            net=getattr(self, 'net'+n)
            net.eval()

    def train(self):
        """
        paired with the self.eval(), switch the flag of the network eval() status flag
        """
        for n in self.model_names:
            net=getattr(self, 'net'+n)
            net.train()

    def load_pretrained_layers(self, name, load_path):
        """
        load pretrained layers from input pth file.
        associate according to layer names.
        name is sub model name in self.model_names, e.g. G, D, Enc, Dec
        """
        net = getattr(self, 'net'+name)
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model %s from %s'%(name, load_path))

        load_state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(load_state_dict, '_metadata'):
            del load_state_dict._metadata
       
        # copy all possible state_dict
        own_state_dict = net.state_dict()
        for l_name, param in load_state_dict.items():
            if l_name not in own_state_dict:
                print('! %s is not in the current state dict'%(l_name))
            else:
                print('copy %s'%(l_name))
                own_state_dict[l_name].copy_(param.data)
