import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks
from collections import OrderedDict, namedtuple
from torch.nn import functional as nnf


class BaseRunner(ABC):
    """

                                BaseRunner
                                /    |    \
                            /        |         \
                        /            |              \
            BaseTrainer     specialModelRunner     BaseValidator
                    \      /                   \   /
                    trainer                   validaor

        combine Model, Loss, Data
        1. receive a model
        2. maintain a data flow
            return results, e.g. calculate a loss/acc
            optimization model parameters if necessary
        5. publish internal status
            maintain visuals, monitoring metrics

        for visual/metrics there are groups of functions
        the basic 3-step process is
            register names
            init the values
            publish current values to other modules(observer)


    usage:
        to design a new base runner for sepcific experiment
        1. update the set_input
        2. design the run functions, and main loss/metrics
        3. register the metrics/visuals 

    dev log:
        refactoring code by separating model with trainer/validator
        trainer and validator are sufficiently different to each other, so that we can have two separate classes

        expose all the dynamic variables to trainer/validator, except for the model parameters or static designs.
        including: input data, output data, latent variables 
        But currently, the dynamic variables are hold in the model class, we need to refactor that

        normally, for each model, there will be a single base runner.
        we can have a training runner/ validation/ test runner based on that base runner if necessary.

    function:
        1. run the model dynamic. Separate data flow from model class. Allow different training dynamics and validation dynamics.
        2. talk to the Director class, send notation to observer. Can be combined with other runners in the single Director flow
        3. each model part running function should be here, because they depend on the input shape

    Usage:
        0. write the base runner. Update the train runner /validator runner if necessary. it will be a multi-inherent structure
            0.1 this is because that some metrics/losses are useful for both train/val, but optimizers are only involved in training. So the baseRunner collect the common metrics/losses, but also allow trainer/validator to update them 
        1. init the metrics
        2. design the loss
        3. run the opimization

    TODO:
        1. colloect the training opttions, or interface configurations
        2. separate the observer from the model as well
        3. separate the scheduler too.
    """
    def __init__(self, opt, tag) -> None:
        """
        we'd better give tag name at init function, so that we can save some code

        tag: unique name tag of each runner. TODO check the uniqueness of the tags
        """
        super().__init__()
        self.opt=opt
        self.tag=tag
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def whoami(self):
        """easier for logging"""
        return self.__class__.__name__+'.'+self.tag

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

    def setup_runner(self, director):
        """
        receive Director and Model after self-init

        so that the trainer can config more 
        because sometimes the model depends on some preprocess steps. 
        """
        self.director=director # the only outsider to talk with
        self.model=director.model
        self.init_displayImages()    
        self.init_monitorMetrics()  

   # ===================== monitoring metrics
    def get_current_losses(self):
        """
        TODO: this should be renamed as publish_current_losses, to make it clear that this should be called by another object, and expose the inner state to them.

        copied from the BaseModel class, so that runners's loss can be fetched by the 
        Return traning losses / errors. train.py will print out these errors on console, and save them to a file
        """
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret
    
    def init_monitorMetrics(self):
        """
            have two paired methods:
            init_x: provide a wrapper 
            register_x: can be re-write by child classes

            avoid hooks? or if we keep the order in the main init method, we can make sure that all register functions are called?
        """
        # own metrics.
        self.loss_names = []

        self.register_monitorMetrics()
        # init here, to allow validation before training
        for l in self.loss_names:
            setattr(self, 'loss_'+l, 0)
        return     

    @abstractmethod
    def register_monitorMetrics(self):
        """

        common metrics, for each runner, they can use tag to distinguish each other
        
        Do not add metircs here unless you are very sure that all the runners need to monitor these metrics 

        only add in the loss_names
        specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>

        the tricky part is that
        - [ not necessary if we are using the tensorboard ]use averaged mini-batch loss for training
        - use total averaged loss for validation/testing

        TODO:
        1. rearrange the metrics according to tensorboard convention
        2. can we use registration pattern instead of using naming?
        3. can we split this big method into smaller ones? by inherent, hooks 
        """
        # tag=self.tag
        # self.loss_names.append('ELBO_'+tag)
        # self.loss_names.append('RMSEx_'+tag)
        # self.loss_names.append('distortion_x_'+tag)

        # for v in ['mu', 'tau']:
        #     for w in ['hi2zi', 'hj2zj']:
        #         self.loss_names.append(v+'_'+w+'_norm_'+tag)
        # for v in ['avgKL_zi', 'avgKL_zj']:
        #     self.loss_names.append(v+'_'+tag) # note, do not add 'loss' with the names

        #return
        pass 

    # ===================== monitoring visuals 
    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        # TODO: what if we want to add lazy plot?
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret
    
    def init_displayImages(self):
        self.visual_names = [] # done int base model init 
        self.register_displayImages()
        return 

    @abstractmethod
    def register_displayImages(self):
        """
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        """
        # tag=self.tag
        # if self.opt.sw_useImgx:
        #     self.visual_names.append('img_x_'+tag)
        #     self.visual_names.append('img_x_hat_clamp_'+tag)
        # return 
        pass

    # ================ input data    
    @abstractmethod
    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        expected input:
            4D tensor [batch(batch_size), domains_index(per_item_domain_num), sample_index(per_domain_point_num), domain_label+class_label+features(14)]
            domain_index is the random index inside the batch, not the global domain_label 

            enc_input_list: because we take some modals as missing. The default is to use all modals. The caveat is that the target domain d is differet with the source domain labels, so we can not use one-hot vector to represent them both.
        
        available options:
        #if self.opt.sw_augment_sigma_x>0:
        #    self.x = self.__augment_data(self.x, self.opt.sw_augment_sigma_x)
 
        """
        # data = input['d'].to(self.model.device)
        # self.data = data # used by the test script to load the data
        # ## the order matters, that is why we need single element in the batch only
        # self.cbi, self.cbj, _= data.shape
       
        # # CAUTION! the naming and shape is important 
        # # 1. ~~it is easier to operate without the batch dim. ~~
        # # 2. we can fix the batch size to 1, but expanding the per_domain_point_num. change the data loading settings in the config
        # # 3. we also ignore the memory speed issues here for now
        # # 4. we use the fact that torch linear layers take arbitary dims [N,*,D]. Change the views for loss functions if needed, or vice versa?
        # # 5. assuming all domains are aligned----each domain has the same number of samples
        # # Conclusion:
        # #   all the variables are organized as 2D (so no need to change loss functions) 
        # #   but viewed as [domain_index, sample_index, features] when it is convenient
        # features = data[:, :, self.meta_feature_dim:] # bi=batch, bj

        # self.x = features.view(self.cbi*self.cbj, self.opt.image_channel, self.opt.image_hw, self.opt.image_hw)
       
        # # TODO: allow hook and obersers here
        # with torch.no_grad():
        #     if self.opt.sw_useImgx:
        #         setattr(self, 'img_x_'+self.tag, self.x[:self.cbj, :].reshape(self.cbj, self.opt.image_channel, self.opt.image_hw, self.opt.image_hw))
        pass

    # =================== some common loss functions 

    def init_torchLossFunc(self):
        """
        pytorch need to init
        define loss functions, for monitor in test as well
        """
        self.criterionBCE = torch.nn.BCELoss()
        self.criterionCE = torch.nn.CrossEntropyLoss()
        self.criterionMSE = torch.nn.MSELoss()
        self.criterionL1 = torch.nn.L1Loss()

    # ================ TODO: remove these functions, because it is not very common
    
    def aneal_scheme(self, scheme):
        """
        allow anealed parameters

        we set it here instead of the trainer, because the validator need to access the settings during train-val loop as well
        """
        if scheme[0] != scheme[1]:
            ## GLL_tau is the precision, 1/variance, increase along the training
            # if standard diviation Sigma is 5 in the start, precision Tau is 1/25
            # if Sigma reduce to 1 in the end, precision Tau is 1
            # if Sigma reduce to 0.1 in the end, precision Tau is 100
            # this will increase the weight of MSE, reducing the relative weight of the KL essentially
            # the problem of balancing reconstruction and KL is that the reconstruction error can be scaled arbitarily?
            # at least according to the dimenions?
            # Sigma = 5 is meaningless if the input x is in [-1, 1]
            # Sigma should be something like 0.1 in the beginning, and increase to 0.01 or so
            # that means the Tau starts from 1e2, increase to 1e4
            # while we keep an eye on the learning rate
            # NO! we reduce the Tau from large to small, allowing learning to reconstruct firt and then KL goes up, meaning sigma is small to big
            # the final sigma should be around 1/50 for our current scaling [7, 54] to [0, 1], meaning Tau is 2500, so we can start from 1e5  
            
            ## aneal GLL tau [0] start bigger tau, [1] end smaller tau, [2] total steps by epoch, [3] repeat?, [4] stop repeating
            # no matter [0], [1] relative, we go from [0] to [1]  
            gt0 = scheme[0]
            gt1 = scheme[1]
            gt2 = scheme[2]
            gt3 = scheme[3]

            step = (gt1-gt0)/gt2

            if gt3 == 0: # no repeat
                target = gt0 + step*self.current_epoch 
            elif scheme[4] <= self.current_epoch:
                pass
            else:
                target = gt0 + step*(self.current_epoch % gt2)

            if ( step<0 and  target <= gt1) \
                    or (step>0 and target >= gt1):
                target = gt1

        else:
            target = scheme[0]

        return target


    def warm_up_by_epoch(self, epoch):
        """
        ask for epoch information from training script, to warm-up the learning 
        """
        #if epoch <= self.opt.warmup_epoch+self.opt.pretrainDis_epoch:
        if epoch <= self.opt.warmup_epoch:
            self.current_Gibbs_sweeps = 0
        else:
            self.current_Gibbs_sweeps = 0#self.opt.Gibbs_sweeps

        self.current_epoch = epoch
        if hasattr(self.opt, 'aneal_GLL_tau'):
            self.GLL_tau = self.aneal_scheme(self.opt.aneal_GLL_tau)
            self.loss_GLL_tau = self.GLL_tau
        
        if hasattr(self.opt, 'aneal_beta_KLzj'):
            self.beta_KLzj = self.aneal_scheme(self.opt.aneal_beta_KLzj)
            self.loss_beta_KLzj = self.beta_KLzj

        if hasattr(self.opt, 'aneal_beta_KLzi'):
            self.beta_KLzi = self.aneal_scheme(self.opt.aneal_beta_KLzi)
            self.loss_beta_KLzi = self.beta_KLzi
        return 