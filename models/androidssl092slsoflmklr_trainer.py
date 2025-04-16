import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import model_utils

import numpy as np

from .base_trainer import BaseTrainer
from .androidssl01base_runner import androidssl01baseRunner

from ginvae.util.named_tuple import concatenate_namedtuples

# from .ILD_alpha import alpha_to_LR

from .kemar_env import kemar_env
from .stochastic_lso import sample_lso_array

class androidssl092slsoflmklrtrainer(BaseTrainer, androidssl01baseRunner):
    """
    092slsoflmklr

    1. assert single LSO  
    2. sample the lso, rather than using the rate

    -----

    091lsoflmklr 

    replace the angle comparison with the realy LSO
    
    -----
    flmklr: Full-360-range Limited Motion Kemar Left-right
    based on 05constrainlsoildtrainer

    updates:
    1. switch to kemar with 360 range
    2. use LR detector instead real label

    ----------
    05 constrain lso

    limite the range of the rotation to -30, 30
    within the range, allow rotation, use LSO-gradient learning
    out the range, stop at the boundary, use bootstrapping-gradient based on the current model via calculation

    time 0:
        input angle y
        predict angle y1
    time 1:
        rotate to r
        calculate new input, predict angle y2
        then y=r+y2
    true loss is abs(y-y1)
    but y is not available, replace it as (r+y2)
    so the sorrogate loss is abs(y-r-y2)
    ---------
    A trainrunner is a trainer+runner

    we have options
        1. train Order and CPT predicitions separately
        2. train them altogether
        3. merge them after a while 

    """
    def __init__(self, opt, tag) -> None:
        super().__init__(opt, tag)

        # ESC_file=os.path.join(self.opt.dataroot, 'ampnorm_merged_train.wav')

        ESC_file=None

        if self.opt.binaural_ratios[0]=='EMPTY':
            self.env = kemar_env(fluc_level=self.opt.fluc_level, fluc_elev=[float(x) for x in self.opt.fluc_elev], sound_lib=ESC_file) # shared env for all batch
        else:
            self.env = kemar_env(ratio_coeff=[float(x) for x in self.opt.binaural_ratios], fluc_level=self.opt.fluc_level, fluc_elev=[float(x) for x in self.opt.fluc_elev], sound_lib=ESC_file)

        self.rolling_loss = model_utils.MovingAverage() # 100 is the window size
        self.rolling_loss_angle = model_utils.MovingAverage() # 100 is the window size
        self.rolling_loss_reg = model_utils.MovingAverage() # 100 is the window size
        self.rolling_loss_total = model_utils.MovingAverage() # 100 is the window size


        self.lso_idx=[]
        if self.opt.lso_idx_list[0]=='EMPTY':
            self.lso_idx=list(range(32))
        else:
            for x in self.opt.lso_idx_list:
                self.lso_idx.append(int(x))
        assert len(self.lso_idx)==1 or len(self.lso_idx)==32, 'lso_idx_list should be 1 or 32. Otherwise code is wrong!!!'

    def create_dataset(self):
        #pass
        # TODO refactor
        N=self.opt.max_dataset_size
        self.dataset= np.ones(N)#None
        self.dataset_size= N # a max number

    def register_monitorMetrics(self):
        """
        """
        
        # type: bezier curve type
        # L2y: L2 loss of the average position

        #for m in ['ACCh2o', 'LLh2o', 'LLx2cpts']:
        for m in ['rmse','total','reg','angle']:
            self.loss_names.extend([m+'_'+self.tag])

        return 

    def optimize_parameters(self, total_iters):
        
        with AccumulatedOptimizeNetContextManager(self.model, self.model.model_names, self, total_iters, self.opt.opt_cycle):

            angle = self.s_angle = np.random.uniform(0, 360)
            # angle = self.s_angle = np.random.uniform(-90, 90)
            # if angle<0:
            #     angle=angle+360 # 270 to 360
            sarb=self.env.sample_angle_a_sound(angle)
            input_array=np.concatenate((sarb.r_levels, sarb.l_levels), axis=0)
            input_array=input_array/100.0 # scale down the numbers
            y_pred_logit = self.model.netandroid(torch.from_numpy(input_array).float()) # default 24*2
            #y_pred=2*torch.sigmoid(y_pred)-1 # scale to [-1, 1]
            y_pred=4*torch.sigmoid(y_pred_logit)-2 # scale to [-2, 2], make it easier to learn with the sigmoid function. Sigmoid output should be around [0.25, 0.75]

            # angle_sin=torch.sin(torch.tensor([angle/180*np.pi], dtype=torch.float)).to(y_pred.device)
            # angle_cos=torch.cos(torch.tensor([angle/180*np.pi], dtype=torch.float)).to(y_pred.device)

            # loss = torch.mean((y_pred[0]-angle_sin)**2+ (y_pred[1]-angle_cos)**2)

            with torch.no_grad():
                # # calculate the predicted angle by converting the output sin and cos to angle in degree
                # # y_pred_angle = torch.atan2(y_pred[1], y_pred[0])*180/np.pi
                # # clip the cos and sin to [-1, 1]
                # cos_a=torch.clamp(y_pred[1], -1, 1)
                # y_a = torch.acos(cos_a) # [0, pi], -1->pi, 1->0
                # # convert to [0, 2pi]
                # y_a = y_a*180/np.pi
                # # change the range to [0, 360], given the sin value
                # # if y_pred[0]>0.5:
                # if y_pred[0]>=0:
                #     y_a = y_a
                # else:
                #     y_a = 360-y_a

                y_a = torch.atan2(y_pred[0], y_pred[1])*180/np.pi
                # convert from -180 to 180 to 0 to 360
                y_a = y_a if y_a >= 0 else y_a + 360

            # # -------- set the simple input environment

            # #s_angle = np.random.uniform(-90, 90)
            # s_angle = np.random.uniform(-5, 5)
            # #s_angle = np.random.uniform(-10, 10)

            # scaled_angle = (s_angle+180)/360 # [-90, 90] will be [0.25, 0.75]
            # #s_ILD = alpha_to_I(s_angle, SOUND_FREQUENCE)
            # r_input, l_input = alpha_to_LR(s_angle)
            
            # #s_ILD = torch.tensor([r_input, l_input], dtype=torch.float) # reuse the name s_ILD, but for batch=1 only
            # s_ILD = torch.tensor([0, 0, r_input-l_input], dtype=torch.float) # reuse the name s_ILD, but for batch=1 only
            
            # # _, c = sess.run([optimizer, rmse_loss], feed_dict={x: np.reshape(s_ILD, [1, 2]),
            # #                                         y: np.reshape(scaled_angle, [1, 1]),
            # #                                         real_y: np.reshape(scaled_angle, [1, 1])})

            # y_pred = self.model.netandroid(s_ILD.reshape(1, 3))
            # y_pred = torch.sigmoid(y_pred)

            # record the training performance, base on the first prediction
            with torch.no_grad():
                
                d1=torch.abs(y_a-angle)
                d=torch.minimum(d1, 360-d1)
                mse = torch.sqrt(d**2)

                avg_l=self.rolling_loss.update(mse.item())
                # TODO: make this automated. Small mistakes such as '_' format cause bugs, otherwise.
                setattr(self, 'loss_rmse_'+self.tag, avg_l) # no need to take sum, we just want to see the average 

            # define rotation-range
            # rr=(0.416, 0.583) # [-30, 30] will be scaled to [0.416, 0.583]
            # rr=(0.347, 0.513) # [-5, 5]
            rr=[330, 30] # [-30, 30] # 
            rr=[0, 360]
            # if torch.abs(angle-y_a)<2: #
            #         loss=y_pred_logit[0]*0
            # elif y_a>=rr[0] or y_a<=rr[1]: 
            if y_a>=rr[0] or y_a<=rr[1]: 
                # mimic_lso = torch.sign(y_pred-scaled_angle).detach()
                # # gradient of the abs() function
                # loss = mimic_lso*y_pred # the gradient decrease the loss

                # find out the relative direction


                # # TODO: move into a function, and check the input range
                # # anything beyond [0, 360] will give silent error!!!
                # if y_a>=angle: # only works for positive angles
                #     # because if y_a=360, angle=-90, then d1=450>360, again need to rotate backward
                #     d1=y_a-angle # +
                #     d2=360+angle-y_a # +
                #     # d1+d2=360
                #     if d1<d2: # d1<180
                #         rotate=-1 # overshot, backward
                #     else:
                #         rotate=+1
                # else: # y_a<=angle # 
                #     d1=angle-y_a
                #     d2=360-(angle-y_a)
                #     if d1<d2:
                #         rotate=1 # undershot, forward
                #     else:
                #         rotate=-1
                
                # ----- now we use LSO to decide the rotation direction
                # update the sound and angle
                scaled_action = (y_a+180)/360.0
                sarb1=self.env.transit_wo_reward(scaled_action)
                ild_array1=sarb1.r_levels-sarb1.l_levels # right - left
                # HERE the back/front does not really matter
                lso_array_sample=sample_lso_array(self.env.freqs, ild_array1, idx_list=self.lso_idx)[:, 1]


                # lso_output = self.model.netlsoreader(torch.from_numpy(lso_array_sample).float()) # return a float in [0, 1]
                # lso_output = torch.sigmoid(lso_output) # scale to [0, 1]
                # lso_sign = torch.sign(lso_output-0.5) # fire or no fire, left 1, right -1
                if len(self.lso_idx)==1: 
                    lso_output=torch.tensor(lso_array_sample[0])
                else:
                    lso_output = self.model.netlsoreader(torch.from_numpy(lso_array_sample).float()) # return a float 
                    lso_output = torch.sigmoid(lso_output) # scale to [0, 1]
                lso_sample = torch.bernoulli(lso_output).item() # fire or not
                lso_sign = torch.sign(torch.tensor(lso_sample-0.5)) # fire or no fire, left 1, right -1
                
                rotate = -lso_sign
                
                y_dummy = y_a+rotate*10 # 10 degree for numerical safety
                # loss = torch.abs(y_dummy-y_pred)
                angle_sin=torch.sin(torch.tensor([y_dummy/180*np.pi], dtype=torch.float)).to(y_pred.device)
                angle_cos=torch.cos(torch.tensor([y_dummy/180*np.pi], dtype=torch.float)).to(y_pred.device)

                angle_loss = torch.abs(y_pred[0]-angle_sin) + torch.abs(y_pred[1]-angle_cos)
                reg= torch.abs(y_pred[0]**2+y_pred[1]**2-1)
                loss = angle_loss+reg*1.0

                with torch.no_grad():
                    angle_loss_l=self.rolling_loss_angle.update(angle_loss.item())
                    # TODO: make this automated. Small mistakes such as '_' format cause bugs, otherwise.
                    setattr(self, 'loss_angle_'+self.tag, angle_loss_l) # no need to take sum, we just want to see the average 
                    # resiger the regularity loss and total loss
                    reg_l=self.rolling_loss_reg.update(reg.item())
                    setattr(self, 'loss_reg_'+self.tag, reg_l)
                    total_l=self.rolling_loss_total.update(loss.item())
                    setattr(self, 'loss_total_'+self.tag, total_l)

            else:
                # with torch.no_grad():
                #     # move to the maximum range
                #     if y_pred>rr[1]:
                #         rotate=rr[1]
                #     else:
                #         rotate=rr[0]
                #     scaled_angle1 = rotate-scaled_angle
                #     # keep the range of the input within [-90, 90]
                #     # ignore the error caused by wrong summation
                #     if scaled_angle1>0.75:
                #         scaled_angle1=0.75
                #     if scaled_angle1<0.25:
                #         scaled_angle1=0.25
                #     angle1=scaled_angle1*360-180
                #     r_input, l_input = alpha_to_LR(angle1)
                #     s_ILD = torch.tensor([0, 0, r_input-l_input], dtype=torch.float) # reuse the name s_ILD, but for batch=1 only
                #     y_pred1 = self.model.netandroid(s_ILD.reshape(1, 3))
                #     y_pred1 = torch.sigmoid(y_pred1) 
                #     total_guess = y_pred1+rotate

                # loss = torch.abs(y_pred-total_guess)
                pass

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