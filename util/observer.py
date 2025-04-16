from torch.utils.tensorboard import SummaryWriter
import os
import time
class Observer():
    """
    this is a temporary solution for replacing the Visdom visualiser. First step to updating the design patterns
    
    we align the methods with Visualiser, so that we can replace it easily in the code.

    TODO
    [ ] reuse the print loss
    [ ] replace logging with logging package
    [ ] replace Visdom to async Tensorboard
    [ ] allow multiple observers
    [ ] use hierarchical directory structure as required by tensorboard
    [ ] return the whole history? vs allow flexible hooks for diagnosis? 
    """

    def __init__(self, opt):
        self.opt = opt  # cache the option
        self.display_id = opt.display_id

        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)
    
        self.log_dir=os.path.join(opt.checkpoints_dir, opt.name) 
        self.tb_writer=SummaryWriter(log_dir=self.log_dir)
        self.init_time=time.time()

        # to separate the tensorboard plot
        # TODO: should be passed in. should not need to check
        self.metric_group=['tau_', 'mu_', 'grad_norm', 'RMSEx', 'LLy', 'ACCy', 'avgKL', 'distortion', 'SVI_', 'SVI_q', 'zj2yi', 'ELBO']

    def display_current_results(self, visuals, epoch, save_result, filetag=''):
        """
        TODO: remove unnecessary arguments to adhocly aligned with the visdom version
        """
        # raise NotImplementedError
        for label, images in visuals.items():
            self.tb_writer.add_images(label, images)        
    
    def reset(self):
        pass # not used anymore

    def plot_current_losses(self, epoch, counter_ratio, losses, total_iter):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
            total_iter (int)      -- not used by visdom, for consistency with tensorboard observer 
        
        TODO:
            how to avoid recording total iter? global iter?
        """

        for k, v in losses.items():
            flag_find_g=0
            for g in self.metric_group:
                if g in k:
                    flag_find_g=1
                    self.tb_writer.add_scalar(g+'/'+k, v, global_step=total_iter, walltime=time.time()-self.init_time) 
                    #self.tb_writer.add_scalars(g, {k:v}, global_step=total_iter, walltime=time.time()-self.init_time) 
            if not flag_find_g:
               self.tb_writer.add_scalar(k, v, global_step=total_iter, walltime=time.time()-self.init_time) 
        #self.tb_writer.add_scalars('Observer0', losses, global_step=total_iter, walltime=time.time()-self.init_time)

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        # TODO add self name tag in printing 
        # TODO printing to files?

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = 'Observer: (epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %+.3e ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
