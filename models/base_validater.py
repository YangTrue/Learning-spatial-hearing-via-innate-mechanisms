import torch
import copy
from ginvae.data import create_dataset
from abc import ABC, abstractmethod
from . import model_utils
from .base_runner import BaseRunner


class BaseValidater(BaseRunner):
    # """

    #                                 BaseRunner
    #                                 /    |    \\
    #                             /        |         \\
    #                         /            |              \\
    #             BaseTrainer     specialModelRunner     BaseValidator
    #                     \      /                   \   /
    #                     trainer                   validaor


    #     independent validater, to be combined with a runner 

    #     validater does not allow callback during val process. 
    #     it only returns a final result.

    #     it calls set_input()
    #     which only defined in Runner class or a combined class

    #     tasks:
    #     1. [x] create_evaluation_dataset    
    #     2. [x] opt.eval_epoch_freq
    #     3. [x] actually generate dataset
    #     4. [ ] implement model.validate()

    #     improve
    #     1.[x] do not repeat dataset creation
    #     2.[ ] limit the eval dataset size, random subset. But we are reusing the train opt here for once. no way to shuffle on the fly?
    #     3.[ ] eval more frequently than at most once per epoch
    # """

    def __init__(self, opt, tag) -> None:
        super().__init__(opt, tag)
  
        if self.tag in ['val', 'test', 'tdv']:
            self.val_repeat_range = self.opt.val_repeat_sampling
        else:
            self.val_repeat_range = 1
        
        #opt.phase = 'train'
        self.batch_size = opt.batch_size

        self.create_dataset()

    def create_dataset(self):
        """
        prepare evaluation dataset.
        we make the dataset preparation independent from the Director here, so that a Runner is a complete experiment setting, can run without Director

        tag choose from 'val' 'tdv' (training_domain_val)
        # TODO: we do not update this tag selection thing for the whole project
        but we can re-write this method during visualisation validation
        """
        opt = copy.deepcopy(self.opt) # otherwise changes on opt will be propagated outside
        opt.phase = self.tag # temperally overload the phase option. Therefore share all the dataset options with training, except path prefix "dataroot/phase/". Recover the phase value to 'train' after making the dataset.
        #if phase_code == 'val' or phase_code=='test':
        #if True:
        #    opt.batch_size = 1 # allow flexible bj for each item
        self.dataset = create_dataset(opt)
        self.dataset_size = len(self.dataset)
        print('The number of %s images = %d' % (self.tag, self.dataset_size))
        return 

    def validate(self):
        """
        run the model,
        get the accumulated averge loss for the validation dataset
        carry on that values in printing, in the following training printing
        but not plotted
        """
        model=self.model
        self.phase_code=self.tag
        # TODO: progress bar for infinite dataset size?
        print('<<< enter %s procedure:%d'%(self.phase_code, self.dataset_size))
        # very important, otherwise Torch create graph for each input, give out of Memory errors
        # the grad flag can be turned back on if needed
        with torch.no_grad():     
            model.eval()# TODO: keep the orginal state turn-off dropout and batch-norm
            original_model_phase_code=model.phase_code
            model.phase_code=self.phase_code

            print("datasetsize %d, batchsize %d"%(self.dataset_size, self.batch_size))
            self.init_validate() # set loss to zero for each new validation cycle
            
            for val_repeat in range(self.val_repeat_range):
                print('repeat%d'%val_repeat, end='', flush=True)
                if val_repeat>=1:
                    # because there is uncertainty for single sample, we do multiple repeat
                    # but do not do this randomization if there is no need of repeat sampling
                    self.dataset.val_restart()

                for data in model_utils.progressbar(self.dataset, step=self.batch_size):  
                    self.set_input(data) # unpack data from dataset and apply preprocessing
                    # for doge, we add the phase code here to distinguish val and tdv
                    self.validate_batch()   # accumulate loss/acc values
                    
                    # #if (i*self.batch_size*100)%self.dataset_size==0: # this pregress bar is wrong!
                    # percent_progress=max(int(self.dataset_size*1.0/100/self.batch_size),1)
                    # if (i*self.dataset_size)%percent_progress==0:
                    #     print('.', end='', flush=True)
                self.summarize_per_val_repeat()    
            #if True:# display images on visdom and save images to a HTML file
            #    save_result = 1
            #    model.compute_visuals()
            #    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result, filetag='val')
            
            #model.new_save_visuals(train_epoch, nrow)
                        
            # restore the model phase_code
            model.phase_code=original_model_phase_code

        print('leave '+self.tag+' validation procedure >>>')
        return self.summarize_val_results() 

    def test(self):
        """
        overwrite the base
        Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.get_loss()
            
            for i in range(len(self.y)): # for all samples in the batch
                self.total_sample += 1
            # TODO: need to check this dimensions
            self.total_batch += 1
    
    @abstractmethod
    def init_validate(self):
        """
        reset validate loss to zero
        the loss will be kept as previous val results in the following training procedure print
        """
        pass

    @abstractmethod
    def validate_batch(self):
        """
        run a single batch
        no need to return any thing
        """
        pass
    
    @abstractmethod
    def summarize_val_results(self):
        """
        called at the end of validator.validate()
        returned results will be used by Director to compare mdoels.

        output:
            r -- an OrderedDict of result. The first one is the most important result, used by the director to decide if we need to save the model or not.
        """
        # r = []
        # for yk in ['y']:
        #     #if self.opt.sw_regression:
        #     if False:
        #         track = getattr(self, 'track_valMSE'+yk)
        #         accyk = -track.inspect().item() # get python float value out of tensor
        #     else:
        #         track = getattr(self, 'track_valACC'+yk)
        #         accyk = track.inspect()#.item() # get python float value out of tensor
        #     r.append(accyk)
        # r.append(statistics.mean(r))
        # return r
        pass
    
    def summarize_per_val_repeat(self):
        pass