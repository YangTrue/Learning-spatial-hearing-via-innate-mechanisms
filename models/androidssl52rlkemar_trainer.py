import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import model_utils

import numpy as np

from .base_trainer import BaseTrainer
from .androidssl52base_runner import androidssl52baseRunner

from ginvae.util.named_tuple import concatenate_namedtuples

#from .ILD_alpha import alpha_to_LR
# from .ssl_env import ssl_env
from .kemar_env import kemar_env

class androidssl52rlkemartrainer(BaseTrainer, androidssl52baseRunner):
    """

    52rlkemar

    change from previous ssl_env to new kemar environment
    train with ILD and real reward rather than LSO 
    
    # ------ 
    A trainrunner is a trainer+runner

    we have options
        1. train Order and CPT predicitions separately
        2. train them altogether
        3. merge them after a while 

    """

    def __init__(self, opt, tag) -> None:
        super().__init__(opt, tag)
        #self.env = ssl_env(self.opt.env_sigma, self.opt.env_success_reward) # shared env for all batch
        self.env = kemar_env(self.opt.env_sigma, self.opt.env_success_reward, self.opt.env_step_punish) # shared env for all batch
        self.rolling_loss = model_utils.MovingAverage() # 100 is the window size
        
    def register_monitorMetrics(self):
        super().register_monitorMetrics()
        
        for m in ['rreward', 'rsteps', 'rdone', 'poliloss']:
            self.loss_names.extend([m+'_'+self.tag])

        return 

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

            env = self.env # change the name, unnecessary
            #env.reset()
            sarb=env.sample_angle_a_sound() # scaled_angle_rendered_binaual

            cached_rewards = []
            cached_log_prob = []
            gamma = 0.0 # 0.01 # 0.99 # how much do current action affect the future reward?

            for i in range(self.opt.max_rollout_steps):

                ild_array=sarb.r_levels-sarb.l_levels # right - left 
                # scaled_angle, LRinput = env.reformat_state() # rewrite
                LRinput=torch.from_numpy(ild_array.reshape(1, -1)).float()
                action, log_prob = self.action(LRinput) # action is the predicted angle scaled to [0, 1]
                # reward, done = env.transit(action) 
                reward, done = env.transit_w_reward(action[0,0]) # use the ground truth reward

                cached_rewards.append(reward)
                cached_log_prob.append(log_prob[0,0])

                # for loss recording
                scaled_angle=sarb.angle_01

                if done != 0:                    
                    break
                else: # observe the state, no update
                    sarb=env.scale_angle_render_sound() # scaled_angle_rendered_binaual

            # get the accumulated reward for each step (state, action)
            # here we use the sum of all the steps. Another choice is to use the start state V(s0)
            k = len(cached_rewards)

            # ------- mult-step learning
            cached_rewards.reverse()
            cached_log_prob.reverse()
            G = torch.tensor(0, dtype=torch.float, device=action.device)
            policy_loss = torch.tensor(0, dtype=torch.float, device=action.device)
            for r,l in zip(cached_rewards, cached_log_prob):
                G = G*gamma + r
                policy_loss += l * G

            # # ------- last-step learning only
            # policy_loss = cached_rewards[-1]

            
            loss = -policy_loss # the gradient decrease the loss

            with torch.no_grad():
                # TODO algorithm: so far only the final result monitor
                mse = torch.mean((action-scaled_angle)**2)
                # TODO: make this automated. Small mistakes such as '_' format cause bugs, otherwise.
                setattr(self, 'loss_rmse_'+self.tag, torch.sqrt(mse)) # no need to take sum, we just want to see the average 
                setattr(self, 'loss_rreward_'+self.tag, torch.tensor(cached_rewards).sum()) 
                setattr(self, 'loss_rsteps_'+self.tag, len(cached_rewards)) 
                setattr(self, 'loss_rdone_'+self.tag, done) 
                setattr(self, 'loss_poliloss_'+self.tag, policy_loss) 
            
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