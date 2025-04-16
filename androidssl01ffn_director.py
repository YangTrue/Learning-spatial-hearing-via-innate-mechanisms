from collections import OrderedDict
from ginvae.options.train_options import TrainOptions
from ginvae.models import create_module

from ginvae.base_director2 import BaseDirector

class androidssl01director(BaseDirector):
    """
        keep this simple class as an example for future coding.
        actually can merge these into the BaseDirector
    """
    
    def define_runners(self, opt):
        self.runners=[]
        # =========== trainer init the dataset
        if not opt.doge_validate_only:
            self.trainer=create_module(opt, opt.trainer_name, module_type='trainer', tag='train') 
            self.runners.append(self.trainer)
        # =========== Add optional evaluator 
        if opt.eval_epoch_freq>0 or opt.doge_validate_only:
            # TODO do not use val_er to name the modules. Use the commander pattern instead. 
            self.val_er=create_module(opt, opt.validater_name, module_type='validater', tag='val')
            self.runners.append(self.val_er)
        return 

    # def update_epoch_end(self):
    #     """
    #     replace the default director method here
    #     """
    #     if self.opt.eval_epoch_freq > 0: 
    #         if self.epoch % self.opt.eval_epoch_freq == 0:    # print training losses and evaluation losses and save logging infomation to the disk, duplicate the training loss print if possible. At most one eval per epoch.
    #             #del dataset # to free GPU memory. Cost time here
    #             print('=== entering evaluation process, end of epoch')
    #             raw_r=self.val_er.validate()

    #             # r=raw_r 
    #             # self.update_best_val_result(r, 'val', sw_save_model=True)
 
    #             # here the dataset size should not be here
    #             # and the timer is too deeply coupled
    #             self.print_loss(self.iter_start_time, self.opt, self.visualizer, self.epoch, self.epoch_iter, self.t_data, self.dataset_size)
    #             #dataset = create_dataset(opt)

    #     if self.epoch % self.opt.save_epoch_freq == 0:    # cache our model every <save_epoch_freq> epochs
    #         print('saving the model at the end of epoch %d, iters %d' % (self.epoch, self.total_iters))
    #         self.model.save_networks('latest')
    #         #model.save_networks(epoch)

    #     print('End of epoch %d / %d \t Time Taken: %d sec' % (self.epoch, self.opt.niter + self.opt.niter_decay, time.time() - self.epoch_start_time))

    #     self.trainer.update_learning_rate()    # update learning rates at the end of every epoch.

    #     self.epoch+=1
    #     return


    def compare_best_val_result(self, r, r_old):
        """
        here we always return true
        we only use the validation for monitoring purpose, but alway select the last epoch val as the final model
        this is because we are not going to use a true meta-validation set, due to the cross-validation of the domains

        input: 
            r -- an OrderedDictionary, of results, with the main result at the first position

        output:
            decision ---- bool, yes if r is better than previous
            updated_r ---- the better result
        """
        # if r_old is None:
        #     decision=True
        #     updated_r=r
        # else:
        #     k=list(r)[0]
        #     decision=(r[k]>=r_old[k])
        #     if decision:
        #         updated_r=r
        #     else:
        #        updated_r=r_old
        decision=True
        updated_r=r
        return decision, updated_r
    
if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    director = androidssl01director(opt)
    if opt.doge_validate_only:
        result=director.test_only()
    else:
        result=director.train_and_val()
    print(result)
