import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import model_utils

import numpy as np

from .base_trainer import BaseTrainer
from .androidssl01base_runner import androidssl01baseRunner

from ginvae.util.named_tuple import concatenate_namedtuples

from .ILD_alpha import alpha_to_LR


class androidssl01trainer(BaseTrainer, androidssl01baseRunner):
    """
    A trainrunner is a trainer+runner

    we have options
        1. train Order and CPT predicitions separately
        2. train them altogether
        3. merge them after a while 

    """
    def create_dataset(self):
        #pass
        # TODO refactor
        N=self.opt.max_dataset_size
        self.dataset= np.ones(N)#None
        self.dataset_size= N # a max number
        return 
    
    def optimize_parameters(self, total_iters):
        
        with AccumulatedOptimizeNetContextManager(self.model, self.model.model_names, self, total_iters, self.opt.opt_cycle):
            
            # -------- set the simple input environment

            s_angle = np.random.uniform(-90, 90)
            scaled_angle = (s_angle+180)/360 # [-90, 90] will be [0.25, 0.75]
            #s_ILD = alpha_to_I(s_angle, SOUND_FREQUENCE)
            r_input, l_input = alpha_to_LR(s_angle)
            s_ILD = torch.tensor([r_input, l_input], dtype=torch.float) # reuse the name s_ILD, but for batch=1 only

            # _, c = sess.run([optimizer, rmse_loss], feed_dict={x: np.reshape(s_ILD, [1, 2]),
            #                                         y: np.reshape(scaled_angle, [1, 1]),
            #                                         real_y: np.reshape(scaled_angle, [1, 1])})

            y_pred = self.model.netandroid(s_ILD.reshape(1, 2))
            y_pred = torch.sigmoid(y_pred)

            mse = torch.mean((y_pred-scaled_angle)**2)
            #mse = (y_pred-scaled_angle)**2

            loss = mse # the gradient decrease the loss

            with torch.no_grad():
                # TODO: make this automated. Small mistakes such as '_' format cause bugs, otherwise.
                setattr(self, 'loss_rmse_'+self.tag, torch.sqrt(mse)) # no need to take sum, we just want to see the average 
            
            if self.opt.debug_nan:
                with torch.autograd.detect_anomaly():
                    loss.backward()
            else:
                loss.backward()

        return 
    
    def notify_train_end(self):
        # TODO refactor
        if hasattr(self.opt, 'sw_savejit'):
            if self.opt.sw_savejit:
                self.model.save_jit_networks(self.director.epoch)
        self.director.update_train_end()
        return

class AccumulatedOptimizeNetContextManager:
    """
    count the iterations
    accumlate the gradient
    do not optimize the parameters, until steps get to the threshold

    TODO:
        1. what happens in the end? extra memory?
        2. what shall we do about the network name-list? 

    replace the OptimizeNetContextManager class with new manager

    ======== old comments

    TODO refactor to trainer_runner class

    select the networks to optimize.
    allow network freeze and printing
    """
    def __init__(self, model, net_namelist, trainer, global_steps, opt_cycle):
        self.model = model
        self.net_namelist = net_namelist
        self.trainer=trainer

        #global_steps=global_steps-1 # make it starts from 0
        self.sw_opt = (global_steps%opt_cycle==0)
        self.sw_0_step = (global_steps==1) # the very first step

    def reset_nets(self):
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
        return
    
    def __enter__(self):
        """
            TODO: is it correct to re-set the requires_grad?
            doesn't this change the network behaviour?

        """
        
        # NOTE do not do this every time!!
        # otherwise the order of zero_grad() and step() is wrong!
        if self.sw_0_step:
            #zero grad
            self.reset_nets()

        if self.model.opt.debug_nan:
                torch.autograd.set_detect_anomaly(True) 
            
    def __exit__(self, exc_type, exc_val, exc_tb):
    #take grad steps
        if self.sw_opt:
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
                    #torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0) # clip the gradient norm to 1.0, prevent bad mini batch
                    o = getattr(self.trainer, 'optimizer_'+name)
                    o.step()

            # NOTE clear the gradient after taking steps
            self.reset_nets()
                
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