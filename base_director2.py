"""

Log:
    we re-write the director class.
    make a new file, to allow the old code run OK.

    main changes to do:
    1. remove the tdv and extra options for the out-of-domain experiment control flow
    2. start to make changes for the new pattern: 
        director only call the runners, do not make lower-level decisions
        do not assume the function of other objects. the less methods you call, the easier to change the code. all the logics are in the same place
    3. remove the Visdom Visualiser reference 

old comments:

    # todo:
    # 1. jupyter show docstring
    # 2. python string split with out repeat space " "
    # 3. baidu and bing search are garbage

    # the problem is:
    # 1. too many options mixed with the main program
    # 2. the interface between different components/modules/subprocess is not clearly defined, so you worry that a single change will destroy the thing, in the worst case, saliently

"""


from collections import OrderedDict
from ginvae.options.train_options import TrainOptions
from ginvae.models import create_module
from ginvae.util.observer import Observer

import copy
import time
# import torch # Nothing about Torch

import os
import numpy as np
import sys
from abc import ABC, abstractmethod



def cmd_train_helper(base_cmd, **kwargs):
    """
    
    helper function for loop training based on cmd line input string

    TODO! 
    1. the name and  checkpoints_dir option will be used during parse(), has to be specified in the base_cmd string! not the kwargs
    2. the cmd string has to end with \n
    """
    sys.argv = base_cmd.split(' ') # replace empty sys args, to feed into argparse
    opt = TrainOptions().parse()  # get train options
    # note that here we print the train options after parsing the base_cmd, which is not consistent with what is really used
    # specify the options if needed
    for name, value in kwargs.items():
        #opt.training_domains_list = [1, 2, 3]
        # not allowed to use opt[name]
        getattr(opt, name) # doublecheck if name exist
        setattr(opt, name, value)
    return opt

class BaseDirector(ABC):
    """

    230701 picasso

    ========== Old version comments Doge

    Director:
    1. define and link runners
    2. config the hooks during training loop 
        - this is because we need to open the interface for trainer to talk with other runners.
        - a better way is perhaps to directly give trainer the access? Not necessarily? we need a document for analysis
        - director is the center. Talks to everyone. All other component talk to the director only.
        - a better way is to remove the center? Like each layer in the network? But when to evoke what? that will be a flow-based thing?

    Other players(python classes/objects):
    1. networks: model
    2. runners: trainer, validator
    3. helper: monitor, option
    
    the runners will differentiate themselves by names and classies

    we should make this framework more flexible.
    it is a higher level
    compare to the keras or so. 

    update methods can be used for training
    if we only need to run the model, we can use the simple create_models() methods only

    TODO:
    [ ] replace the phase_code to tag, currently it is only used to load the correct dataset
    [ ] the t_data time calculation is not correct or useful?
    
    """
    def __init__(self, opt) -> None:
        """
        the outline of design:
        1. have Model, Runner(Trainer, validater), Director, Observer
        2. Each Director combines one Trainer and multiple validaters, and an Observer to collect and print the loggings.
        3. Each Runner care about 3 things: get a shared model, maintain a private dataset, publish a set of notifications/metrics/visualisations. 
        4. Each model cares only about model defination, save, load and so on.
        5. Observer cares only about where to log, not what/when to log.
        """

        # TODO: naming the functions with init_ to group them better
        # sometimes the model structure depends on the data
        self.init_preprocess_rawdata(opt) # preprocessing of the original dataset, split the files
        self.define_runners(opt) 

        self.update_options(opt) # update self.opt # what is this?

        # TODO: change the flow. Now we use self.opt after updating it. what can we do to improve? make it easier to handle, no need to worry about the option updates anymore? 
        self.define_model(self.opt)
        self.define_observer(self.opt)
        self.init_model_selection()

        # ============ setup runners
        # TODO Python does not follow such function call defination?
        for r in self.runners:
            r.setup_runner(self)

        return 

    #@abstractmethod
    def init_preprocess_rawdata(self, opt):
        """
        
        split dataset into train/test/val/tdv if ncessary
        """
        # #========= Create dataset first
        # if hasattr(opt, 'swon_cm'):
        #     cmfMRI_intraDomain_split(opt)
        # if opt.dataset_mode=='chairs':
        #     chairs_interDomain_split(opt)
        # else:
        #     doge_abidedata_split(opt)
        # return 
        pass
    
    @abstractmethod
    def define_runners(self, opt):
        # # EEEEEEEEEEE define runners
        # self.runners=[]
        # # =========== trainer init the dataset
        # self.trainer=create_module(opt, opt.trainer_name, 'trainer', 'train') 
        # self.runners.append(self.trainer)
        # # =========== Add optional evaluator 
        # if opt.eval_epoch_freq > 0: 
        #     self.val_er=create_module(opt, opt.validater_name, 'validater', 'val')
        #     self.runners.append(self.val_er)
        #     if opt.percent_val_training_domain>0:
        #         # extra validater for training domains
        #         self.tdv_er=create_module(opt, opt.tdv_name, 'validater', 'tdv')
        #         self.runners.append(self.tdv_er)
        # return
        pass 
    
    def update_options(self, opt):
        """
            sometimes the only easy way to update the configure, is to over-write the options.
            this over-writing may based on the dataset preprocessing result.

            # TODO be very careful if you want to re-write it or not
        """

        # # EEEEEEEEEEEEE updates options
        # # convert the personID if there is auxilary task for classification
        # if hasattr(opt, 'swon_auxPersonID'):
        #     if opt.swon_auxPersonID:
        #         if hasattr(self.trainer.dataset.dataset, 'convert_personID'):
        #             n_pid0, n_pid1=self.trainer.dataset.dataset.convert_personID()
        #             # add new attributes to opt
        #             opt.n_pid0=n_pid0
        #             opt.n_pid1=n_pid1
        #         else:
        #             raise NotImplementedError
        # self.opt=opt
        # return 
        self.opt=opt
        pass
    
    def define_model(self, opt):
        """
        TODO refactor
            sometimes model need to be defined later than the runners?
            not really
        """
        # ============= Create model
        model = create_module(opt, opt.model, 'model', 'model')      # create a model given opt.model and other options
        model.print_networks(opt.verbose)

        # TODO: consistent usage of phase_code and tag
        model.phase_code='train'
        self.model=model
        return

    def define_observer(self, opt):
        # ============= create observsers
        if opt.use_tensorboardObserver:
            self.visualizer = Observer(opt)   # create a visualizer that display/save images and plots
        else:
            raise NotImplementedError
            #self.visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
        return 

    def train_and_val(self):
        self.dataset_size=self.trainer.dataset_size
        # =========== main running loop
        self.trainer.run_train_loops() # trainer will call the update_x_timepoint functions

        return self.summarize_results()

    def test_only(self):
        """
        TODO further simplify the code
        simplified running for validation/test only
        perhaps we should write a new class
            - we need to define the val runners? or at least the name of validater/val_er?

        call the validater for once
        the concrete implementation depend on the validater

        """
        self.epoch=0
        # TODO refactor the pattern
        for r in self.runners:
            warm_up_by_epoch = getattr(r, 'warm_up_by_epoch', None) # suppress the getattr exceptions
            if callable(warm_up_by_epoch):
                warm_up_by_epoch(self.epoch)
 
        r=self.val_er.validate() 

        save_result=False
        self.visualizer.display_current_results(self.val_er.get_current_visuals(), self.epoch, save_result)

        losses = self.get_all_current_losses()
        t_comp = 1
        self.visualizer.print_current_losses(1, 1, losses, t_comp, 0)

        # TODO refactoring
        r=(r,0) # make the test_only behave the same as the train_and_val(), returning a tuple of (result, epoch_idx)
        return r

    # 
    #
    # ================= Trainning Hooks, evoked by trainer ===================
    #
    #

    # TODO: what should be managed by the director? what should be added in with diff runners?
    def update_train_begin(self):
        """
        update method for Observer pattern Subscriber. 
        Will be called by the Publisher----trainer. 
        """
        self.epoch=1
        self.total_iters = 0  # the total number of training samples, by datapoint
        self.epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch, by datapoint
        self.t_data = 0 # placeholder for time data, data loading time per data point (normalized by batch_size)

        # =========== Optional evaluation before training
        if self.opt.eval_epoch_freq > 0: 
            self.iter_start_time = time.time()  # timer for computation per iteration

            ## used for warm up in Gibbs Sampling networks
            for r in self.runners:
                warm_up_by_epoch = getattr(r, 'warm_up_by_epoch', None) # suppress the getattr exceptions
                if callable(warm_up_by_epoch):
                    warm_up_by_epoch(self.epoch)
            # print training losses and evaluation losses and save logging infomation to the disk, duplicate the training loss print if possible. At most one eval per epoch.
            #del dataset # to free GPU memory. Cost time here

            # TODO: should iterate through the runners, and they decide what to do by themselves?
            # TODO delete the assumptions about what runners/options do we have

            print('entering validation process, before training')
            self.val_er.validate() 
            # if self.opt.percent_val_training_domain>0:
            #     self.tdv_er.validate() 

            # TODO
            # here the dataset size should not be here
            # and the timer is too deeply coupled
            self.print_loss(self.iter_start_time, self.opt, self.visualizer, self.epoch, self.epoch_iter, self.t_data, self.dataset_size)

        return
 
    def update_epoch_begin(self):
        
        self.epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

        self.epoch_start_time = time.time()  # timer for entire epoch
        self.iter_data_time = time.time()    # timer for data loading per iteration
        self.iter_start_time = time.time()  # timer for computation per iteration
        
        if self.total_iters % self.opt.print_freq == 0:
            self.t_data = self.iter_start_time - self.iter_data_time

        ## used for warm up in Gibbs Sampling networks
        for r in self.runners:
            warm_up_by_epoch = getattr(r, 'warm_up_by_epoch', None) # suppress the getattr exceptions
            if callable(warm_up_by_epoch):
                warm_up_by_epoch(self.epoch)

        return
    
    def update_batch_begin(self):
        self.iter_start_time = time.time()  # timer for computation per iteration
        if self.total_iters % self.opt.print_freq == 0:
            self.t_data = self.iter_start_time - self.iter_data_time

        self.visualizer.reset()
        self.total_iters += self.opt.batch_size
        self.epoch_iter += self.opt.batch_size

        return

    def update_batch_end(self):
        # TODO: this decision should be made by the visualiser. More clear? Allow the visualiser to call the get-all() function
        if self.total_iters % self.opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            save_result = self.total_iters % self.opt.update_html_freq == 0
            #self.model.compute_visuals()
            self.visualizer.display_current_results(self.get_all_current_visuals(), self.epoch, save_result)

        if self.total_iters % self.opt.print_freq == 0:    # print training losses and save logging information to the disk
            self.print_loss(self.iter_start_time, self.opt, self.visualizer, self.epoch, self.epoch_iter, self.t_data, self.dataset_size)

        # TODO This should be moved out into the model object as well        
        if (self.opt.save_latest_freq>0) and (self.total_iters % self.opt.save_latest_freq == 0):   # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (epoch %d, total_iters %d)' % (self.epoch, self.total_iters))
            save_suffix = 'iter_%d' % self.total_iters if self.opt.save_by_iter else 'latest'
            self.model.save_networks(save_suffix)

        self.iter_data_time = time.time()

        return
    
    def update_epoch_end(self):
        if self.opt.eval_epoch_freq > 0: 
            if self.epoch % self.opt.eval_epoch_freq == 0:    # print training losses and evaluation losses and save logging infomation to the disk, duplicate the training loss print if possible. At most one eval per epoch.
                #del dataset # to free GPU memory. Cost time here
                print('=== entering evaluation process, end of epoch')
                # TODO frame refactor
                # the r should be something flexable? the interaface between Director and the Validator
                # so that we can do more flexible things, such as
                # 1. saving the output with book-keepers
                # 2. visualising with the visualiser
                # and we need to allow diff ways to call the validate process as well? Allowing access to global steps and so on?
                # NO good. r should be simple.
                # do not change this Director class. Just pass on the global clock self.epoch
                # Same idea as the printing function. Let the Validator decide if they want to save. 
                #r=self.val_er.validate(epoch=self.epoch) 
                r=self.val_er.validate() # no need the Global clock pass. As the Director is available for the runner to access 
                self.update_best_val_result(r, 'val', sw_save_model=True)

                # TODO if we move this out to to tdv_er, we would be safe to change the runners and options
                # So the key is to avoid calling the runner's method directly, but only notify them
                # if self.opt.percent_val_training_domain>0:
                #     r=self.tdv_er.validate()
                #     self.update_best_val_result(r, 'tdv')
 
                # here the dataset size should not be here
                # and the timer is too deeply coupled
                self.print_loss(self.iter_start_time, self.opt, self.visualizer, self.epoch, self.epoch_iter, self.t_data, self.dataset_size)
                #dataset = create_dataset(opt)

        if self.epoch % self.opt.save_epoch_freq == 0:    # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (self.epoch, self.total_iters))
            self.model.save_networks('latest')
            #model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (self.epoch, self.opt.niter + self.opt.niter_decay, time.time() - self.epoch_start_time))

        self.trainer.update_learning_rate()    # update learning rates at the end of every epoch.
        
        # TODO: move this to specific trianers, no need to leave here
        if not hasattr(self.opt, 'swon_cm'):
            # we do not permute the dataset when using connection matrix
            print("permute the dataset")
            #if not isinstance(self.trainer.dataset, np.array):
            if hasattr(self.trainer.dataset, 'restart'):
                self.trainer.dataset.restart()

        self.epoch+=1
        return
   
    def update_train_end(self):
        print('=== end of training')
        if self.opt.eval_epoch_freq > 0: 
            # if self.opt.percent_val_training_domain>0:
            #     print('Best tdv epoch is %d'%(self.model_selection_results['tdv'][1]))
            print('Best val epoch is %d'%(self.model_selection_results['val'][1]))
        
        #del self.trainer.dataset #explicitly delete, hope that we can cleanup the multiprocessing memory
        
        print('end of training, summary')
        #print('training/val/test domains', end='')
        # print(self.opt.training_domains_list, self.opt.val_domains_list, self.opt.test_domains_list)
    
        if hasattr(self.opt, 'sw_endtest'):
            if self.opt.sw_endtest:
                print('$$$$$$$$$$ start test $$$$$$$$$$$')
                test_validater = create_module(self.opt, self.opt.tester_name, 'validater','test')
                self.model.load_networks('best_val')
                test_validater.validate(self.model, self.epoch, self.opt.display_ncols) 
                self.print_loss(self.iter_start_time, self.opt, self.visualizer, self.epoch, self.epoch_iter, self.t_data, self.dataset_size)
                print('$$$$$$$$$ exit test $$$$$$$$$$$')
        
        #result = {'val_accy':model.track_valACCy.inspect().item()}
        #result = {'val_accy':model.track_valACCy.inspect()}
        result=0
        return result

    def get_all_current_losses(self):
        losses=OrderedDict()
        for r in self.runners:
            l=r.get_current_losses()
            losses.update(l)
        return losses

    def get_all_current_visuals(self):
        """
        we can call compute_current_visuals() in each runner if necessary
        """
        visuals=OrderedDict()
        for r in self.runners:
            l=r.get_current_visuals()
            visuals.update(l)
        return visuals
    
    def print_loss(self, iter_start_time, opt, visualizer, epoch, epoch_iter, t_data, dataset_size):
        """
        print and plot loss
        indexed by training process iter and time.
        used both after each printing period and evaluation phase.

        Parameters:
            epoch (int) -- current epoch
            epoch_iter -- #iter inside this epoch 
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        
        TODO: printing to log files? with name tag? and level?
        """
        losses = self.get_all_current_losses()
        t_comp = (time.time() - iter_start_time) / opt.batch_size
        visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
        if opt.display_id > 0:
            total_iter=epoch*dataset_size+epoch_iter # 1 item=1 iter, 1 batch=multiple iters
            visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses, total_iter)
        return 
    
    def init_model_selection(self):
        # =========== set the best val result
        self.model_selection_results={}
        #for tag in ['val', 'tdv']:
        for tag in ['val']:
            self.model_selection_results[tag]=(None, 0) # result, epoch

    def update_best_val_result(self, r, tag, sw_save_model=False):
        """ customized comparison for validation results to save the best model"""

        decision, updated_r=self.compare_best_val_result(r, self.model_selection_results[tag][0])

        if decision:
            # only update the result and the epoch tag when it is better
            self.model_selection_results[tag]=(updated_r, self.epoch)

            if sw_save_model and self.opt.sw_save_best:
            # used by val validator only, not the tdv validator
                self.model.save_networks('best_val')
        
        return
    
    def compare_best_val_result(self, r, r_old):
        """

        # TODO: here is a hidden decision again
            better to use a tag to make it explicit
        input: 
            r -- an OrderedDictionary, of results, with the main result at the first position

        output:
            decision ---- bool, yes if r is better than previous
            updated_r ---- the better result
        """
        if r_old is None:
            decision=True
            updated_r=r
        else:
            k=list(r)[0]
            decision=(r[k]>=r_old[k])
            if decision:
                updated_r=r
            else:
                updated_r=r_old
        return decision, updated_r
    
    def summarize_results(self):
        return self.model_selection_results['val']

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    director = BaseDirector(opt)
    result=director.train_and_val()
    print(result)
