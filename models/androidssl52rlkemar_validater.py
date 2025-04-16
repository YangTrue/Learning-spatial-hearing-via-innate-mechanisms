from typing import OrderedDict
import torch
import copy
from ginvae.data import create_dataset
from abc import ABC, abstractmethod
from . import model_utils
from .base_validater import BaseValidater
from .androidssl52base_runner import androidssl52baseRunner
import numpy as np
import os

# from .ILD_alpha import alpha_to_LR
from .kemar_env import kemar_env

class Androidssl52rlkemarvalidater(BaseValidater, androidssl52baseRunner):
    """
    52rlkemar

    NOT USED!
    -----------
    improve
    2.[ ] limit the eval dataset size, random subset. But we are reusing the train opt here for once. no way to shuffle on the fly?
    3.[ ] eval more frequently than at most once per epoch
    """
    def create_dataset(self):
        # TODO refactor
        N=self.opt.max_dataset_size
        self.dataset= np.ones(N)#None
        self.dataset_size= N # a max number
        self.env = kemar_env() # shared env for all batch
        return 
    
    def test_actor(self, output_dir):
        """
            instead of sampling the Gaussian 
            we use fixed Mu as the the output action
            we also ignore the SSL environment

            pro: simple, consistent with other testing
            con: not aligned with the code in training
        """
        n_bins = 181
        output_angle = torch.zeros([n_bins,])
        target_angle = torch.zeros([n_bins,])
        
        for i in range(0, n_bins):
            s_angle = i - 90

            r_input, l_input = alpha_to_LR(s_angle, s=55) # fix the sound level to 55
            #s_ILD = [r_input, l_input]
            s_ILD = torch.tensor([r_input, l_input], dtype=torch.float) # reuse the name s_ILD, but for batch=1 only

            y_pred = self.model.netandroid(s_ILD.reshape(1, 2))
            y_pred = torch.sigmoid(y_pred[0, 0])

            output_angle[i] = y_pred
            
            scaled_angle = (s_angle+180)/360 # [-90, 90] will be [0.25, 0.75]
            target_angle[i] = scaled_angle

        mrse = (torch.sqrt((output_angle-target_angle)**2)).mean() 
        unscaled_rmse = mrse*360 
        print('test mean squred angle: {:.3f}'.format(unscaled_rmse))

        #np.save(os.path.join(output_dir, 'huberNet_test_output.npy'), output_angle) # replace the old one

        setattr(self, 'loss_rmse_'+self.tag, unscaled_rmse) # no need to take sum, we just want to see the average 

        epoch = self.director.epoch
        torch.save(output_angle, os.path.join(self.save_dir, 'valid_output_'+str(epoch)+'.pt'))

        return unscaled_rmse
    
    def validate(self):
        """
        run the model,
        get the accumulated averge loss for the validation dataset
        carry on that values in printing, in the following training printing
        but not plotted
        """
        model=self.model
        self.phase_code=self.tag

        # # TODO: progress bar for infinite dataset size?
        # print('<<< enter %s procedure:%d'%(self.phase_code, self.dataset_size))

        # very important, otherwise Torch create graph for each input, give out of Memory errors
        # the grad flag can be turned back on if needed
        with torch.no_grad():     
            model.eval()# TODO: keep the orginal state turn-off dropout and batch-norm
            original_model_phase_code=model.phase_code
            model.phase_code=self.phase_code

            #print("datasetsize %d, batchsize %d"%(self.dataset_size, self.batch_size))
            # self.init_validate() # set loss to zero for each new validation cycle
            
            # for val_repeat in range(self.val_repeat_range):
            #     print('repeat%d'%val_repeat, end='', flush=True)
            #     if val_repeat>=1:
            #         # because there is uncertainty for single sample, we do multiple repeat
            #         # but do not do this randomization if there is no need of repeat sampling
            #         self.dataset.val_restart()

            #     for data in model_utils.progressbar(self.dataset, step=self.batch_size):  
            #         self.set_input(data) # unpack data from dataset and apply preprocessing
            #         # for doge, we add the phase code here to distinguish val and tdv
            #         self.validate_batch()   # accumulate loss/acc values
                    
            #         # #if (i*self.batch_size*100)%self.dataset_size==0: # this pregress bar is wrong!
            #         # percent_progress=max(int(self.dataset_size*1.0/100/self.batch_size),1)
            #         # if (i*self.dataset_size)%percent_progress==0:
            #         #     print('.', end='', flush=True)
            #     self.summarize_per_val_repeat()    
            # #if True:# display images on visdom and save images to a HTML file
            # #    save_result = 1
            # #    model.compute_visuals()
            # #    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result, filetag='val')
            
            # #model.new_save_visuals(train_epoch, nrow)
            
            unscaled_rmse = self.test_actor(self.save_dir)

            # restore the model phase_code
            model.phase_code=original_model_phase_code

        print('leave '+self.tag+' validation procedure >>>')
        
        #return self.summarize_val_results() 
        return unscaled_rmse 

# empty abstract method
    

    def init_validate(self):
        """
        reset validate loss to zero
        the loss will be kept as previous val results in the following training procedure print
        """
        # self.track_valRMSE = model_utils.MovingAverage()
        # self.track_valACCo = model_utils.MovingAverage()
        # self.track_valLLh2cpts = model_utils.MovingAverage()

        # self.raw_acco_total=[] # keep the acc results for each element, instead of batch average. so that we can check the running avg function correctness
        pass

    def validate_batch(self):
        """
        we do not do backward here, so no gradient information
        use phase_code val, tdv(training_domain_val) for doge
        """
    #     r=self.get_suploss_all(self.x)
    #     with torch.no_grad():
    #         # TODO avoid this repeat for train/test and different variables?
    #         avg_valLLo=self.track_valLLo.update(r.LL_h2o.mean())
    #         setattr(self, 'loss_LLh2o_'+self.tag, avg_valLLo) # no need to take sum, we just want to see the average 

    #         avg_valACCo=self.track_valACCo.update(r.bacc_o_hat)
    #         setattr(self, 'loss_ACCh2o_'+self.tag, avg_valACCo)

    #         self.raw_acco_total.append(r.raw_acco.cpu().numpy().reshape(-1,1)) # when batch size is 1, convert from 0-D array to 2D?

    #         # for LL h2cpts, do not take the mean
    #         avg_valLLh2cpts=self.track_valLLh2cpts.update(r.LL_cpts)
    #         setattr(self, 'loss_LLh2cpts_'+self.tag, avg_valLLh2cpts) # no need to take sum, we just want to see the average 
    #     return
        pass

    def summarize_val_results(self):
        """
        output:
            r -- an OrderedDict of result. 
                The first one is the most important result, used by the director to decide if we need to save the model or not. the larger the better.
        
        TODO: use variable name to avoid repeated code
        """
    #     r=OrderedDict()
    #     r['val_acco']=self.track_valACCo.inspect()

    #     t=np.concatenate(self.raw_acco_total)
    #     ri=t.sum()
    #     acc=ri*1.0/len(t)
    #     print(self.whoami()+': validation result summary: ri=h2o, avg acc, len(total)', ri, acc, len(t))
    #     r['val_acco']=acc
    #     return r
        pass

    # def summarize_per_val_repeat(self):
    #     t=np.concatenate(self.raw_acco_total)
    #     ri=t.sum()
    #     acc=ri*1.0/len(t)
    #     print(self.whoami()+': validation per repeat summary: ri=h2o, avg acc, len(total)', ri, acc, len(t))
    #     #print(ri, acc, len(t))
    #     print(self.whoami()+': validation per repeat summary: valACCo', self.track_valACCo.inspect())
    
    # def concatenate_raw(r):
    #     """
    #     when batch-size ==1 there is a bug:
    #         self.raw_acco_total[0]
    #         array(True)
    #         self.raw_acco_total[0].shape
    #         ()
    #         type(self.raw_acco_total[0])
    #         <class 'numpy.ndarray'>
    #         self.raw_acco_total[0].size
    #         1
    #     Error: zero-dimensional arraies can not be concatenated.

    #     0-D arrays, or Scalars, are the elements in an array. Each value in an array is a 0-D array.
    #     An array that has 0-D arrays as its elements is called uni-dimensional or 1-D array.
    #     this is a bug for numpy.concatenate()?

    #     Solution 0711: add dim in the result
    #     """
    #     #if r[0]
    #     pass
