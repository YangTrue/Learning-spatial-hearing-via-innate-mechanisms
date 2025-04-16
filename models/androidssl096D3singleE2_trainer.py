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
from ginvae.models.rotation_x_axis import rotation_x_axis, degree_difference, convert_pred_to_angle

class androidssl096D3singleE2trainer(BaseTrainer, androidssl01baseRunner):
    """
    096D3singleE2

    E2: expanding to 

    we use another network to copy the teacher network's output
    and use the teacher network's output as the target

    should be much easier to learn
    and easier to expand the network capacity if needed

    avoid the difficulties of training a single network: stability and plasticity conflict

    during training 
    1. we load previous trained model both for the android and the teacher. so that the android has a good easy init
    2. the teacher will be fixed, and the android will be trained to copy the teacher's output at the new range.
    this stabalization trick is also used by deep RL, as a weight copy. But different, because the weight copy is off-line learning with correction, but ours are not corrected. 
    we can do correction in between though, not needed.

    instead of copying the logits output, we need to convert to angle and then convert back.
    
    The problem we are facing is actually a multi-task expanding problem, gradually adding in the task. very classic.
    we do not want to make it too complicated here, just use the simple way to solve it, to prove our own point.
    
    -------------------

    096D3single

    1. extend the singel hearing from making 2D prediction to 3D
    2. output chagnes
    3. environment changes
    4. validation changes -- different error measurement
    5. allow training from given range
    

    the peak is at (80-90, 295-180)=(-10, 115)

    there are 3 layers of complexity
    1. the 3D rotation, coordinate system updates
    2. the rotation range is limited
    3. the prediction is relative to the original point, but the rotation is around the peak point

    
    done
    1. change the env to allow contol of elev sample
    2. limit the range of reactions, make the training easier
    
    ----
    095single

    use single ear to listen twice for both direction
    compare the intensity and choose the direction
    it is actually exactly the same
    using a simpler circuit, but slower, need twice listening
    
    -----------
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

        if self.opt.binaural_ratios[0]=='EMPTY':
            self.env = kemar_env(fluc_level=self.opt.fluc_level, fluc_elev=[float(x) for x in self.opt.fluc_elev], sw_elev=True) # shared env for all batch
        else:
            self.env = kemar_env(ratio_coeff=[float(x) for x in self.opt.binaural_ratios], fluc_level=self.opt.fluc_level, fluc_elev=[float(x) for x in self.opt.fluc_elev], sw_elev=True) # shared env for all batch

        self.rolling_loss = model_utils.MovingAverage() # 100 is the window size
        self.rolling_loss_angle = model_utils.MovingAverage() # 100 is the window size
        self.rolling_loss_reg = model_utils.MovingAverage() # 100 is the window size
        self.rolling_loss_total = model_utils.MovingAverage() # 100 is the window size

        self.rolling_loss_elev = model_utils.MovingAverage() # 100 is the window size
        self.rolling_loss_D3 = model_utils.MovingAverage() # 100 is the window size

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
        
        for m in ['elev','D3']:
            self.loss_names.extend([m+'_'+self.tag])

        return 

    def optimize_parameters(self, total_iters):
        
        with AccumulatedOptimizeNetContextManager(self.model, self.model.model_names, self, total_iters, self.opt.opt_cycle):

            sa0, sa1, se0, se1 = self.opt.sound_range
            angle = self.s_angle = np.random.uniform(sa0, sa1) # 0 to 360! NO! can be 115-x, to 115+x, perhaps negative
            elev = self.s_elev = np.random.uniform(se0, se1) # -90 to 90
            # NOTE: if the value is not as assumed, the calculation will be wrong. How to avoid such silient errors?            

            # angle = self.s_angle = np.random.uniform(-90, 90)
            # if angle<0:
            #     angle=angle+360 # 270 to 360
            sarb=self.env.sample_angle_a_sound(angle, elev=elev)
            input_array=np.concatenate((sarb.r_levels, sarb.l_levels), axis=0)
            input_array=input_array/100.0 # scale down the numbers
            y_pred_logit_st = self.model.netandroid(torch.from_numpy(input_array).float()) # default 24*2
            #y_pred=2*torch.sigmoid(y_pred)-1 # scale to [-1, 1]
            y_pred_st=4*torch.sigmoid(y_pred_logit_st)-2 # scale to [-2, 2], make it easier to learn with the sigmoid function. Sigmoid output should be around [0.25, 0.75]

            # angle_sin=torch.sin(torch.tensor([angle/180*np.pi], dtype=torch.float)).to(y_pred.device)
            # angle_cos=torch.cos(torch.tensor([angle/180*np.pi], dtype=torch.float)).to(y_pred.device)

            # loss = torch.mean((y_pred[0]-angle_sin)**2+ (y_pred[1]-angle_cos)**2)
            

            with torch.no_grad():
                # let's get the supervised signal from the teacher network
                y_pred_logit_t = self.model.netteacher(torch.from_numpy(input_array).float()) # default 24*2
                y_pred_t=4*torch.sigmoid(y_pred_logit_t)-2 # scale to [-2, 2], make it easier to learn with the sigmoid function. Sigmoid output should be around [0.25, 0.75] 

                y_a_st, elev_pred_st = convert_pred_to_angle(y_pred_st)
                y_a_t, elev_pred_t = convert_pred_to_angle(y_pred_t)


                # ==== move head ====

                # we are not going to really move here, just use its value?
                # no. we have to move, and listen again, to get the new prediction
                # even though sometimes it is not used --- which is rare

                # the right ear peak (aizm = 115, elev= -10) should be within the range
                peak_azim = 115
                peak_elev = -10
                peak_azim_mid = peak_azim+180 # we fix it here
                # the rotation range is about the absolute origin point
                a0, a1, e0, e1 = self.opt.rotation_range

                if y_a_t > a0 and y_a_t < a1 and elev_pred_t > e0 and elev_pred_t < e1:
                    is_within_range_t = True
                else:
                    is_within_range_t = False
                
                if y_a_st > a0 and y_a_st < a1 and elev_pred_st > e0 and elev_pred_st < e1:
                    is_within_range_st = True
                else:
                    is_within_range_st = False

                if is_within_range_t and is_within_range_st:
                    is_within_range = True
                else:
                    is_within_range = False

                

                if is_within_range:
                    # pass # do nothing

                    # we still move toward the target
                    # because the first guess can be very wrong, this will give it a second chance, a single step searching/correction

                    # folow the teacher's output. No need to worry about the gradient here
                    y_a = y_a_t
                    elev_pred = elev_pred_t

                    r_azim = y_a-peak_azim
                    r_elev = elev_pred-peak_elev

                    new_azim = angle-r_azim # range [-360, 360] here
                    new_azim, new_elev = rotation_x_axis(-r_elev, new_azim, elev) # azim range [0, 360], elev range [-90, 90]

                    # new_angle = angle-rotate # [-270, 270] 
                    # # if the elev reaches the limit, then the new angle should be another direction, change the azimuth sign as well.

                    # ---- listen again, at new position
                    new_sarb=self.env.sample_angle_a_sound(new_azim, elev=new_elev)
                    new_input_array=np.concatenate((new_sarb.r_levels, new_sarb.l_levels), axis=0)
                    new_input_array/=100.0 # scale down the numbers

                    new_y_pred_logit = self.model.netteacher(torch.from_numpy(new_input_array).float()) # default 24*2
                    #y_pred=2*torch.sigmoid(y_pred)-1 # scale to [-1, 1]
                    new_y_pred=4*torch.sigmoid(new_y_pred_logit)-2 # scale to [-2, 2], make it easier to learn with the sigmoid function. Sigmoid output should be around [0.25, 0.75]

                    new_pred_azim, new_pred_elev = convert_pred_to_angle(new_y_pred)
                else:
                    # out of range
                    # listen again to extend the prediction range

                    # ========== decide the target

                    if not is_within_range_t and is_within_range_st:
                        # use the teacher's output
                        # if the teacher decide it is out of range, then it is surely out of range
                        y_a = y_a_t
                        elev_pred = elev_pred_t
                    else:
                        # use the student's output
                        y_a = y_a_st
                        elev_pred = elev_pred_st
                    
                    # ===================== decide the rotaiton

                    # this is wrong, as we need to consider the cycle, which one is closer to the peak?
                    # if y_a <= a0:
                    #     r_azim = a0-peak_azim
                    # if y_a >= a1:
                    #     r_azim = a1-peak_azim
                    # consider the cycle of the azimuth angle
                    # y_a in [0, 360]
                    if y_a >= peak_azim_mid or y_a <= a0:
                        # on the left side, rotate to the left boudary
                        r_azim = a0-peak_azim
                    elif y_a < peak_azim_mid and y_a >= a1:
                        # note hte AND condition
                        # on the right side, rotate to the right boudary
                        r_azim = a1-peak_azim
                    else:
                        # within the azimuth range
                        r_azim = y_a-peak_azim
                    
                    if elev_pred <= e0:
                        r_elev = e0-peak_elev
                    #if elev_pred >= e1: # WRONG!!!
                    # then the r_elev will be re-write by the else again!
                    elif elev_pred >= e1:
                        r_elev = e1-peak_elev
                    else:
                        # within the elevation range
                        r_elev = elev_pred-peak_elev

                    # ============== calculate the new location relative to the origin point ==============

                    # get the new angle
                    # we only consider the cycle here, because we know that the rotation range is really small.
                    # will they be handled automatically by the slab Env?
                    # corner case 1: elev is at the limit. 
                    # corner case 2: azimuth is at the limit
                    # corner case 3: both are at the limit
                    # easy, get it done easily

                    # ------------------
                    # case 1:
                    # ------------------
                    # if elev_pred+r_elev>90: should be (a-r)-90, rather than a-r.
                    # if elev_pred+r_elev<-90: should be (a-r)+90, rather than a-r.

                    # ------------------
                    # case 2:
                    # ------------------

                    new_azim = angle-r_azim # range [-360, 360] here
                    # new_elev = elev-r_elev

                    # followed by another x-axis rotation
                    new_azim, new_elev = rotation_x_axis(-r_elev, new_azim, elev) # azim range [0, 360], elev range [-90, 90]

                    # new_angle = angle-rotate # [-270, 270] 
                    # # if the elev reaches the limit, then the new angle should be another direction, change the azimuth sign as well.

                    # ---- listen again, at new position
                    new_sarb=self.env.sample_angle_a_sound(new_azim, elev=new_elev)
                    new_input_array=np.concatenate((new_sarb.r_levels, new_sarb.l_levels), axis=0)
                    new_input_array/=100.0 # scale down the numbers

                    new_y_pred_logit = self.model.netteacher(torch.from_numpy(new_input_array).float()) # default 24*2
                    #y_pred=2*torch.sigmoid(y_pred)-1 # scale to [-1, 1]
                    new_y_pred=4*torch.sigmoid(new_y_pred_logit)-2 # scale to [-2, 2], make it easier to learn with the sigmoid function. Sigmoid output should be around [0.25, 0.75]

                    new_pred_azim, new_pred_elev = convert_pred_to_angle(new_y_pred)


            # skip the two-side test
            with torch.no_grad():

                # ========= decide the target =========
                # do not separate the azimuth and elevation, because the rotation is done together
                if is_within_range:
                    # # directly copy the teacher's output
                    # dummy_y_azimuth = y_a 
                    # dummy_y_elev = elev_pred

                    # still allow a correction, this may not be stable? but should be OK?
                    azim_2, elev_2 = rotation_x_axis(r_elev, new_pred_azim, new_pred_elev) # reverse the x-axis rotation
                    azim_1 = azim_2+r_azim # reverse the z-axis rotation
                    elev_1 = elev_2

                    # following the teacher, but still double-check it
                    if degree_difference(y_a_t, elev_pred_t, azim_1, elev_1) <= 3:
                        # passed the double check, no need to re-learn
                        # keep it stable, just in case
                        dummy_y_azimuth = y_a
                        dummy_y_elev = elev_pred
                    else:
                        # failed the double check
                        dummy_y_azimuth = azim_1
                        dummy_y_elev = elev_1
                else:
                    azim_2, elev_2 = rotation_x_axis(r_elev, new_pred_azim, new_pred_elev) # reverse the x-axis rotation
                    azim_1 = azim_2+r_azim # reverse the z-axis rotation
                    elev_1 = elev_2

                    dummy_y_azimuth = azim_1
                    dummy_y_elev = elev_1


            # ====================
            # ========= loss calculation =========
            # ====================
                    
            # calculate the loss use the dummy_y
            # record the training performance, base on the first prediction
            # use 3D angle calculation                    
            # monitor the azimuth and elevation separately as well, just in case for debugging

            # loss = torch.abs(y_dummy-y_pred)
            # now we need to shift the target back to the original point
            # note that this is because we called the convert_pred_to_angle function, which is based on the peak point
            angle_sin=torch.sin(torch.tensor([np.deg2rad(dummy_y_azimuth-115)], dtype=torch.float)).to(y_pred_st.device)
            angle_cos=torch.cos(torch.tensor([np.deg2rad(dummy_y_azimuth-115)], dtype=torch.float)).to(y_pred_st.device)

            angle_loss = torch.abs(y_pred_st[0]-angle_sin) + torch.abs(y_pred_st[1]-angle_cos)
            reg= torch.abs(y_pred_st[0]**2+y_pred_st[1]**2-1)
            azim_loss = angle_loss+reg*1.0
            
            # here is not about rotation, it is just about an init bias
            # so we do not need to rotate, only need to shift
            elev_loss = torch.abs(y_pred_st[2]-(dummy_y_elev+10)/90) # scale to [-1, 1]
            # elev_loss = torch.abs(y_pred[2]-(dummy_y_elev)/90) # scale to [-1, 1]

            loss = azim_loss + elev_loss # similar range of loss

            with torch.no_grad():
                angle_loss_l=self.rolling_loss_angle.update(angle_loss.item())
                # TODO: make this automated. Small mistakes such as '_' format cause bugs, otherwise.
                setattr(self, 'loss_angle_'+self.tag, angle_loss_l) # no need to take sum, we just want to see the average 
                # resiger the regularity loss and total loss
                reg_l=self.rolling_loss_reg.update(reg.item())
                setattr(self, 'loss_reg_'+self.tag, reg_l)


                total_l=self.rolling_loss_total.update(loss.item())
                setattr(self, 'loss_total_'+self.tag, total_l)

                elev_l=self.rolling_loss_elev.update(elev_loss.item())
                setattr(self, 'loss_elev_'+self.tag, elev_l)

            # calculate the 3D angle distance between y_a, y_elev and the true angle
            with torch.no_grad():

                degree_difference_l = self.rolling_loss_D3.update(np.abs(degree_difference(y_a_st, elev_pred_st, angle, elev)))
                setattr(self, 'loss_D3_'+self.tag, degree_difference_l)

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