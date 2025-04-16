import torch
from torch.nn import functional as nnf
import numpy as np
import os

from .base_model import BaseModel
from . import networks_doge as networks
from . import model_utils

from collections import OrderedDict, namedtuple
import statistics

class Androidssl02rlModel(BaseModel):
    """ 

    1. use Gaussian policy
    2. make interactive learning
    
    ----------

    main changes 
    1. instead of loading existing dataset, only use interactive environment
    2. much simpler model structure
    3. more advanced plotting and saving for observation
    
    
    """
    
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """

        Parameters:
            parser          -- original option parser
            [not used  is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        
        TODO:
            2. can we manage the options better? their relationships?
                2.1 check their compatability
                2.2 detect the un-used options. Visualise the reference modules
                2.3 manage options by groups. Automatically add. e.g. lr_x2hj for network lr_x2hj
            3. turning this into another basemodel with VAE, our own basemodel
            
        """
        # --------------------- model specific options 

        parser.add_argument('--android_mlp_lw', nargs='+', type=str, default=['EMPTY'], help='width mlp encoder')

        parser.add_argument('--max_rollout_steps', type=int, default=1, help='max steps allowed in a single rollout')   
        parser.add_argument('--env_sigma', type=float, default=40, help='sigma for the env bell-shape reward curve, hardness of the env')   
        parser.add_argument('--env_step_punish', type=float, default=-0.1, help='sigma for the env bell-shape reward curve, hardness of the env')   
        parser.add_argument('--env_success_reward', type=float, default=100.0, help='reward for solving the env. Matters as the variance scale')   
        
        parser.add_argument('--policy_sigma', type=float, default=0.01, help='exploration sigma for gaussian policy')   

        #  -------------------- default necessary options

       # changing the default values 
        parser.set_defaults(norm='none')
        parser.set_defaults(netG=None)

        # parser.add_argument('--image_hw', type=int, default=64, help='image width and height, assume they are the same')
        # parser.add_argument('--image_channel', type=int, default=3, help='image channel')
        # #parser.add_argument('--deconv_hw', type=int, default=64, help='input for the deconv network, width and height, assume they are the same')
        # #parser.add_argument('--deconv_channel', type=int, default=3, help='image for the decov network, channel')
        # #parser.add_argument('--output_pixel_range', type=str,default='-11', help='select the range of pixel output. choose from 01 or -11')
        
        parser.add_argument('--sw_useImgx', type=int, default=0, help='switch for visualise image x and x_hat, disentangled with sw_useST. meaning that we take in image data')   

        # parser.add_argument('--x2h_conv_lw', nargs='+', type=str, default=['EMPTY'], help='width mlp encoder')
        #parser.add_argument('--h2order_mlp_lw', nargs='+', type=str, default=['EMPTY'], help='width mlp encoder')
        parser.add_argument('--midline_lsomidline_lw', nargs='+', type=str, default=['EMPTY'], help='width mlp encoder')

        # parser.add_argument('--load_x2h', type=str, default='', help='laoding pretrained path')
        # parser.add_argument('--load_h2order', type=str, default='', help='laoding pretrained path')         

        parser.add_argument('--opt_cycle', type=int, default=1, help='accumulate gradient inside a cycle. 5 means taking operater steps after accumulating 5 steps gradients')
       
        # parser.add_argument('--LLh2cpts_weight', type=float, default=1.0, help='weight for regularizing the loss of regression term')  
        # #parser.add_argument('--aneal_GLL_tau', nargs='+', type=float, help='anealing scheme for GLL tau')

        parser.add_argument('--load_net_common_dir', type=str, default='', help='shared common directory for loading networks. Otherwise you can also load different nets from diff paths')
        parser.add_argument('--load_net_common_prefix', type=str, default='', help='shared common directory for loading networks. Otherwise you can also load different nets from diff paths')
        # parser.add_argument('--noload_zj2yDec', action='store_true', help='do not load network')

        parser.add_argument('--debug_nan',action='store_true' , help='debug nan, slow')

        #TODO: update framework design
        parser.add_argument('--use_tensorboardObserver',action='store_true' , help='use tensorboard instead of visdom for more stable async training and logging. Ad hoc option here')

        # TODO configure these options properly, otherwise the Train option is different from the Test
        parser.add_argument('--doge_validate_only',action='store_true' , help='doge validate only, different from the orginal is_train switch. used by the director')
        # # TODO because this option is cited by the base runner, we have to keep it here. Not good, how to remove?
        parser.add_argument('--warmup_epoch', type=int, default=0, help='EXTRA epoch on top of the pretraining epoch number of warmup without using gibbs sweeps')   
        return parser

    def __init__(self, opt, tag):
        """Initialize the class.

        Parameters:
            opt (Option class, a python namespace)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        
        if self.isTrain:
            self.phase_code = 'train'
        else:
            self.phase_code = 'test'

        self.setup_config(opt)
        self.init_networks(opt)
        self.init_GPUplaceholder(opt)    

    def setup_config(self, opt):
        """
        placeholder function, double check and update the opt settings
        translate some configurations into model attributes
        """
        #assert self.opt.batch_size == 1 # for data loading process, convinient for contrastive learning
        # define networks
        #self.meta_feature_dim = 3 # define the format
        #self.feature_dim = self.opt.item_dimension-self.meta_feature_dim # first 3, i, j, y
        
        #assert self.opt.zizjzk2x_lw[-1] == self.feature_dim # we fix the sigma here 
        #assert self.opt.x2hj_lw[0] == self.feature_dim #+  self.dim_z_i
 
        return 

    def init_networks(self, opt):
        """
        specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        
        # use convention: 
        #   1. netname_type_lw e.g. netname_mlp_lw
            2. load network
        # TODO: add option naming check. the bad thing is that we can not easily find the network by names in the code.
        #e.g. self.model_names = ['zizjzk2x','zj2y', 'x2hj', 'hj2zj', 'hi2zi', 'x2hi', 'x2zk'] #, 'zi0'
        """
        nettype2fun={'mlp':networks.define_MLPEncoder, 'mlpleaky':networks.define_MLPEncoder, 'conv':networks.define_FullConvNet, 'permuinvamlp':networks.define_permuinvaMLPEncoder, 'convunet':networks.define_ConvUNet, 'lsoreadout':networks.define_LSOReadout, 'lsomidline':networks.define_LSOMidline}
        self.model_names=[]
        for k in opt.__dict__:
            if '_lw' in k: # pickup the lw(layer width) settings
                ## check emtpy networks
                # TODO how to control the option logics? select only one of the types?               
                kopt=getattr(opt, k)
                if kopt[0]=='EMPTY':
                    print("empty network for %s"%(k))
                    continue

                ## register the network
                netname=k.split('_')[0]
                nettype=k.split('_')[1]
                self.model_names.append(netname)

                if nettype in nettype2fun.keys():
                    kopt=getattr(opt, k)
                    # TODO: add names
                    #if nettype in ['mlp', 'permuinvamlp']:
                    if 'mlp' in nettype: 
                        kopt=[int(x) for x in kopt]
                    # mlp already use LeakyReLU for default
                    # if 'leaky' in nettype:
                    #     non_linear_type=''
                    net=nettype2fun[nettype](kopt, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
                    # add nettype attributes
                    net.nettype=nettype # or we can use setattr
                    setattr(self, 'net'+netname, net)
                else:
                    raise NotImplementedError('unknown network type in '+k)

                n=netname
                if hasattr(self.opt, 'load_'+n):
                    # load single net before using the common dir
                    fn=getattr(self.opt, 'load_'+n)
                    if fn!='':
                        print(f'loading {n} from {fn}')
                        fn=os.path.join(self.opt.load_net_common_dir, fn) # so that we do not need to change the root in cmd line?
                        self.load_pretrained_layers(n, fn)
                        continue # next network, skip the loading below
                
                if hasattr(self.opt, 'noload_'+n):
                    # skip loading before using the common dir
                    print('skip loading net '+n)
                    continue
                elif self.opt.load_net_common_dir != '':
                    #fn = getattr(self.opt, 'load_net'+n)
                    if self.opt.load_net_common_prefix == '':
                        #prefix = 'latest'
                        prefix = 'best_val'
                    else:
                        prefix = self.opt.load_net_common_prefix
                    fn = prefix+'_net_'+n+'.pth'
                    path = os.path.join(self.opt.load_net_common_dir, fn)
                    if path != '' and fn != '':
                        self.load_pretrained_layers(n, path)
                    else:
                        raise ValueError('unknown file '+path)
        return
 
    def init_GPUplaceholder(self, opt):
        """
        optional
        """
        #self.bi_sites = self.opt.batch_size
        #max_x_len = self.bi_sites*self.opt.bj_persons_per_site*self.opt.bk_frames_per_person
        #max_z_width = self.opt.zizjzk2x_lw[0]
        #max_shape = (max_x_len, max_z_width)
        #self.GPU_ones = torch.ones(max_shape, dtype=torch.float).detach().to(self.device)
        #self.GPU_zeros= torch.zeros(max_shape, dtype=torch.float).detach().to(self.device)
        self.GPU_one = torch.tensor([1.0]).to(self.device)
        self.GPU_zero = torch.tensor([0.0]).to(self.device)

    def tau_positive(self, m, t):
        """
        constrain the value of both m and t
        this is necessary for us becuase the network input is not well constrained for net like xzi2zk. If zi is too big, then zk is doomed. And also bad for learning x and zi together.
        we guess that: even with gradient norm clip, it still has problem, because the norm can concentrate on a single dimension, out of 128 or even higher dimensions
        """
        #m = nnf.tanh(m)*10
        m = m
        t = nnf.softplus(t)+1e-9
        return m, t

    def reparameterize(self, mu, tau):

        std = torch.sqrt(1.0/(tau+1e-9))
        eps = torch.randn_like(std)

        z = eps * std + mu 
	
        return z
 
    def _split_z(self, code):
        """Helper: split the latent dimension into mu and var^{-1}"""
        l2 = code.shape[1]
        l = int(l2/2)
        return code[:, :l], code[:, l:]
    
    def Matrix_GaussianPoE(self, e_mu, e_tau, use_normal_prior, keepdim=True):
        """
        a matrix version

        tau is the "sum" of {expert tau-s}
        mu is the "tau-weighted mean" of {mu-s}
        Both these two operations can be done on-line, or iteratively
        Each point is an expert as well, we also allow multiple expert.s

        INPUTs:
        e_mu: expert mu tensor [n_experts in subgraph, n_domains out of aggregation, n_ppd for aggretation subgraphs, n_factors]
        e_tau: expert var^{-1} tensor 
        use_normal_prior: boolean
        
        OUTPUTs:
        mu: [1, n_domains, 1, n_factors] # keep the dim for easier use
        tau:

        Example:
        for z_y there may be  [2, domains_per_batch x points_per_domain, 1,                     3], i.e each point as a domain
        for z_d there will be [1, domains_per_batch,                     points_per_domain    , 3]

        """
        # the prior
        #tau = torch.ones(  (e_mu.shape[0], 1, e_mu.shape[2]) , dtype=torch.float).to(self.device)
        #mu  = torch.zeros( (e_mu.shape[0], 1, e_mu.shape[2]) , dtype=torch.float).to(self.device)

        if use_normal_prior:
            tau = 1.0 + e_tau.sum(dim=(0, 2), keepdim=keepdim)
        else:
            tau = e_tau.sum(dim=(0, 2), keepdim=keepdim)
        
        weighted_mu = e_mu * e_tau
        #sum_w_mu = mu + weighted_mu.sum(dim=1)
        sum_w_mu = weighted_mu.sum(dim=(0, 2), keepdim=keepdim)
        mu = sum_w_mu / tau

        return mu, tau


       