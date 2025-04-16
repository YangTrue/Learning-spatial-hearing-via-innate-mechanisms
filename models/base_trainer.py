import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import model_utils
from ginvae.data import create_dataset
from . import networks_doge as networks
from .base_runner import BaseRunner

class BaseTrainer(BaseRunner):
    """
                                 BaseRunner
                                /    |    \
                            /        |         \
                        /            |              \
            BaseTrainer     specialModelRunner     BaseValidator
                    \      /                   \   /
                    trainer                   validaor

        a trainer template to handle 
            1. all the optimization details.
        but leave these to runner:
            1. monitoring metrics/visualisations
            2. dataflow
            3. shorthands for getting the loss function

        will be inherited by a trainer, combined with base runner

        this is a Publisher class in the Observer pattern. 
    """
    def __init__(self, opt, tag) -> None:
        super().__init__(opt, tag)
        self.create_dataset() # TODO: revise the process here. Dataset should be separated from the trainer

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
        super().setup_runner(director)
        self.init_optimizers()
        return 

    def init_optimizers(self):
        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.optimizers=[]
        for n in self.model.model_names:
            net_n = getattr(self.model, 'net'+n) 
            self.__opt_registration(net_n, n)

        self.schedulers = [networks.get_scheduler(optimizer, self.opt) for optimizer in self.optimizers]

        return 
    
    def __opt_registration(self, n, name):
        """
        register the network and the name

        protocal: 
            if there is lr
        """
        if hasattr(self.opt, 'lr_'+name):
            lr = getattr(self.opt, 'lr_'+name)
            if lr <= 0:
                lr = self.opt.lr
                print('using default lr %s %f'%(name, lr))
            else:
                print('using lr %s %f'%(name, lr))
        else:
            lr = self.opt.lr
            print('using lr %s %f'%(name, lr))

        o = torch.optim.Adam(n.parameters(), lr=lr, betas=(self.opt.beta1, 0.999))
        setattr(self, 'optimizer_'+name, o)
        self.optimizers.append(o)
        return o
    
    def create_dataset(self):
        opt=self.opt
        # create a new dataset
        # TODO: move the dataset creation to training runner. And return dataset information.
        # the tricky part is that the model need to be init when we init the runner?
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataset)    # get the number of images in the dataset.
        print('The number of training samples = %d' % dataset_size)
        self.dataset=dataset
        self.dataset_size=dataset_size
        return 

    def run_train_loops(self):
        """
        main function for the framework
        """
        self.notify_train_begin()
        opt=self.opt
        # ============== Main trianing loop
        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            self.notify_epoch_begin()

            self.model.train() # validator will turn off with model.eval() each time
            for i, data in enumerate(self.dataset):  # inner loop within one epoch

                self.notify_batch_begin()

                # TODO refactor
                self.set_input(data)         # unpack data from dataset and apply preprocessing
                self.optimize_parameters(self.director.total_iters)   # calculate loss functions, get gradients, update network weights

                self.notify_batch_end()
       
            self.notify_epoch_end()
        self.notify_train_end()
        return
    
    # define the Observer interfaces
    def notify_train_begin(self):
        self.director.update_train_begin()
        return
 
    def notify_train_end(self):
        self.director.update_train_end()
        return

    def notify_epoch_begin(self):
        self.director.update_epoch_begin()
        return
   
    def notify_epoch_end(self):
        self.director.update_epoch_end()
        return

    def notify_batch_begin(self):
        self.director.update_batch_begin()
        return
    
    def notify_batch_end(self):
        self.director.update_batch_end()
        return 
    
    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        # TODO makes this more explicit, perhaps plot it
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr) 

    @abstractmethod
    def optimize_parameters(self, total_iters):
        """
            the very core process for the trainer function
        """
        pass

class OptimizeNetContextManager:
    """
    TODO refactor to trainer_runner class

    select the networks to optimize.
    allow network freeze and printing
    """
    def __init__(self, model, net_namelist, trainer):
        self.model = model
        self.net_namelist = net_namelist
        self.trainer=trainer
         
    def __enter__(self):
        """
            TODO: is it correct to re-set the requires_grad?
            doesn't this change the network behaviour?

        """
        #zero grad
        for name in self.net_namelist:
            o = getattr(self.trainer, 'optimizer_'+name)
            o.zero_grad()
            if hasattr(self.trainer.opt, 'lr_'+name):
                lr = getattr(self.trainer.opt, 'lr_'+name)
                if lr == 0: # freeze the network learning
                    n = getattr(self.model, 'net'+name)
                    for pname, para in n.named_parameters():
                        para.requires_grad = False
        
        if self.model.opt.debug_nan:
                torch.autograd.set_detect_anomaly(True) 
            
    def __exit__(self, exc_type, exc_val, exc_tb):
    #take grad steps
        for name in self.net_namelist:
            
            net = getattr(self.model, 'net'+name)
            
            # TODO: allow observer hooks here
            setattr(self.trainer, 'loss_grad_norm2_'+name, self.net_grad_norm(name))
            
            flag_train=True
            if hasattr(self.trainer.opt, 'lr_'+name):
                lr = getattr(self.trainer.opt, 'lr_'+name)
                if lr == 0: # freeze the network learning
                    flag_train=False

            if flag_train:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0) # clip the gradient norm to 1.0, prevent bad mini batch
                o = getattr(self.trainer, 'optimizer_'+name)
                o.step()
                
    def net_grad_norm(self, name):
        """
        get the average grad norm for the network parameters 
        """
        net = getattr(self.model, 'net'+name)
        norm = 0
        np = 0
        for p in net.parameters():
            np = np+1
            if p.grad is None:
                # we do not use extra test on the learning rate of the network  
                norm -= 1
                break
            else:
                norm = norm+p.grad.data.norm(2)
        norm = norm/np
        return norm