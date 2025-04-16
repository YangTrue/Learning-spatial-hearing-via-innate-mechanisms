import numpy as np
import torch
class RunningAverage:
    """
    ~~no used if we use the tensorboard.~~

    a running average buffer for a scalar variable.
    used for smoothing the loss or acc log

    it is better to have instant value for debuging, rather than avg self.avgLLy = model_utils.
    """
    def __init__(self, buffer_size):
        self.buffer = np.zeros(buffer_size)
        self.idx = 0 # a circular pointer
        self.valid = 0

    def put(self, v):
        with torch.no_grad():
            self.buffer[self.idx] = v
            self.idx += 1
            if self.valid < len(self.buffer):
                self.valid += 1
            if self.idx == len(self.buffer):
                self.idx = 0
        return self.inspect()

    def inspect(self):
        with torch.no_grad():
            if self.valid < len(self.buffer):
                return np.sum(self.buffer)*1.0 / self.valid
            else:
                return np.mean(self.buffer)


class MovingAverage:
    """
    DO NOT USE THIS!
    1. do not handle the tensor value well
    2. using list is not efficient
    3. wrong average value from init

    Note that this is not the same as the running average, which is a buffer of fixed size.
    This is a moving average, which will be influenced by the old value for infinitely long time.
    e.g. in the case of training loss, the moving average will be influenced by the initial value of the loss, which is not the case for the running average.
    so we should prefer the running average for the loss monitoring.

    what if N is too big?
    
    calculate the average on the fly, without large memory
    Q: can we estimate the numerical precision?
    https://stackoverflow.com/questions/28820904/how-to-efficiently-compute-average-on-the-fly-moving-average


    """
    def __init__(self):
        self.N = 1.0
        self.avg = 0.0
        
        # make a FIFO buffer for the moving average
        self.buffer=[] 

    def update(self, new_v):
        self.avg = self.avg + (new_v-self.avg)/self.N
        self.N += 1

        # update the buffer
        self.buffer.append(new_v)
        # if the buffer is too long, remove the oldest value
        if len(self.buffer) > 100:
            # self.avg = self.avg - (self.buffer[0]-self.avg)/self.N
            self.buffer.pop(0)
            # self.N -= 1
        return self.avg

    def inspect(self):
        return self.avg

class SetNetNoGrad:
    """
    backup the original flag of using grad or not

    so that even in evaluation runs, we can still use grad-based SVI
    """
    def __init__(self, model, net_namelist):
        self.model = model
        self.net_namelist = net_namelist
        
        self.original_status = []
        for name in self.net_namelist:
            n = getattr(self.model, 'net'+name)
            for pname, para in n.named_parameters():
                self.original_status.append(para.requires_grad)

    def __enter__(self):
        # with context https://preshing.com/20110920/the-python-with-statement-by-example/
        for name in self.net_namelist:
            n = getattr(self.model, 'net'+name)
            for pname, para in n.named_parameters():
                para.requires_grad = False
 
    def __exit__(self, exc_type, exc_val, exc_tb): 
        idx = 0 # TODO use dictionary instead of list? what if there are multiple para? At least we should check the name
        for name in self.net_namelist:
            n = getattr(self.model, 'net'+name)
            for pname, para in n.named_parameters():
                para.requires_grad = self.original_status[idx]
                idx = idx+1

class OptimizeNetContextManager:
    """
    TODO refactor: move this to trainer_runner class

    select the networks to optimize.
    allow network freeze and printing
    """
    def __init__(self, model, net_namelist):
        self.model = model
        self.net_namelist = net_namelist
         
    def __enter__(self):
    #zero grad
        for name in self.net_namelist:
            o = getattr(self.model, 'optimizer_'+name)
            o.zero_grad()
            if hasattr(self.model.opt, 'lr_'+name):
                lr = getattr(self.model.opt, 'lr_'+name)
                # TODO refactor: just remove the netname from the list
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
            setattr(self.model, 'loss_grad_norm2_'+name, self.net_grad_norm(name))
            
            flag_train=True
            if hasattr(self.model.opt, 'lr_'+name):
                lr = getattr(self.model.opt, 'lr_'+name)
                if lr == 0: # freeze the network learning
                    flag_train=False

            if flag_train:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0) # clip the gradient norm to 1.0, prevent bad mini batch
                o = getattr(self.model, 'optimizer_'+name)
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


import sys
import time
def progressbar(it, prefix="progressbar", size=60, file=sys.stdout, step=1):
    """
    input:
        it: the iterator
        prefix: leading text
        size: length of the progress bar, how many . or # appears on the screen
        step: sometimes (like the pytorch dataset, each enumerate yields more than 1 data from the iterator)
    #https://stackoverflow.com/questions/3160699/python-progress-bar

    TODO learn better example from https://pythonrepo.com/repo/rsalmei-alive-progress-python-generating-and-working-with-logs
    using the with() pattern, and clearer setting
    """
    count = len(it)
    t0=time.time()
    def show(j):
        x = int(size*j/count)
        t1=time.time()

        time_elapsed=int(t1-t0)
        ratio=j*1.0/count
        if ratio==0:
            eta=-1
        else:
            eta=(t1-t0)/ratio
        
        file.write("%s[%s%s] %i/total-%i-steps %i/eta-%i-secs\r" % (prefix, "#"*x, "."*(size-x), j, count, time_elapsed, eta))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i*step+1)
    file.write("\n")
    file.flush()