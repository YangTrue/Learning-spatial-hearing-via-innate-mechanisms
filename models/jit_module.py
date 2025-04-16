import torch
import numpy as np
import os
from collections import OrderedDict, namedtuple
from torch.nn import functional as nnf


class Jitmodule():
    """
    class to 
    1. load trained jit modules from given path
    2. specify the input output dimension
    3. combine the sub-networks into a single module, so that from the outside, this is a single function

    no need to record the acc
    
    """

    @staticmethod
    def named_modify_commandline_options(name, parser, is_train=True):
        """
        called by runners

        Parameters:
            name            -- self name, will be the pre-fix so that 
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        Behaviour:
            add requirements for 
                1. paths of the model
                2. specific requirements
        """

        parser.add_argument('--jitpath_'+name, type=str, default='', help='path for the jit model')  
        parser.add_argument('--yidx_'+name, type=int, default=2, help='index for the output index')

        return  parser


    def __init__(self, name, opt):
        """
        """
        self.name=name
        self.yidx=getattr(opt, 'yidx_'+name)

        srcpath=getattr(opt, 'jitpath_'+name)
        
        # load the jit model
        self.net_x2hj=torch.jit.load(os.path.join(srcpath,'101_jitnet_x2hj.pth'))
        self.net_x2hj.eval()
        self.net_hj2yj=torch.jit.load(os.path.join(srcpath,'101_jitnet_hj2yj.pth'))
        self.net_hj2yj.eval()

        # compose the model if needed
        #self.net=torch.nn.Sequential(self.net_x2hj, self.net_hj2yj) # failed here. WHY? Ah we need to reshape it?
    
    def get_batch_acc(self, x, yi, cbi, cbj):
        """
        assume single sample during evaluation

        TODO remove this assumption
        """

        #=========================# add loss for classification
        x2hj=self.net_x2hj(x)
        x2hj=nnf.relu(x2hj)
        # reshape the convnet output
        logits_zi2yi=self.net_hj2yj(x2hj.view([cbi*cbj,-1]))
        
        #logits_zi2yi=self.net(x)
        
        q_zi2yi = torch.distributions.categorical.Categorical(logits=logits_zi2yi)
        LL_zi2yi = q_zi2yi.log_prob(yi.view(cbi*cbj)) 
        LL_zi2yi = LL_zi2yi.view([cbi, cbj])

        with torch.no_grad():
            #loss_LLzi2yi = LL_zi2yi.mean()
            yi_hat = torch.argmax(logits_zi2yi, dim=1, keepdim=True)
            rawacc_yi=self.classification_rawacc(yi, yi_hat) # calculate the raw 
            bacc_yi=self.classification_acc(yi, yi_hat) # acc for this batch 
        r=namedtuple('result', ['LL_zi2yi','yi_hat', 'bacc_yi','rawacc_yi'])
        return r._make([LL_zi2yi, yi_hat, bacc_yi, rawacc_yi])    

    @staticmethod
    def classification_acc(y_true, y_hat):
        return (y_hat.squeeze().type(torch.uint8) == y_true.squeeze().type(torch.uint8)).type(torch.float).mean()
    
    @staticmethod
    def classification_rawacc(y_true, y_hat):
        return (y_hat.squeeze().type(torch.uint8) == y_true.squeeze().type(torch.uint8)).type(torch.float)