import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
#from . import networks
from collections import OrderedDict, namedtuple
from torch.nn import functional as nnf
from argparse import Namespace
from ginvae.util.named_tuple import concatenate_namedtuples


from ginvae.models.base_runner import BaseRunner
#class dogenew55baseRunner(BaseRunner):
class androidssl01baseRunner(BaseRunner):
    """
    only use the logging parts

                                BaseRunner
                                /    |    \
                            /        |         \
                        /            |              \
            BaseTrainer     specialModelRunner     BaseValidator
                    \      /                   \   /
                    trainer                   validaor


    function:
        1. run the model dynamic. Separate data flow from model class. Allow different training dynamics and validation dynamics.
        2. talk to the Director class, send notation to observer. Can be combined with other runners in the single Director flow
        3. each model part running function should be here, because they depend on the input shape

    Usage:
        0. write the base runner. Update the train runner /validator runner if necessary. it will be a multi-inherent structure
            0.1 this is because that some metrics/losses are useful for both train/val, but optimizers are only involved in training. So the baseRunner collect the common metrics/losses, but also allow trainer/validator to update them 
        1. init the metrics
        2. design the loss
        3. run the opimization
    """

    def register_monitorMetrics(self):
        """

        common metrics, for each runner, they can use runner's own tag to distinguish each other
        
        Do not add metircs here unless you are very sure that all the runners need to monitor these metrics 

        only add in the loss_names
        specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>

        the tricky part is that
        - [ not necessary if we are using the tensorboard ]use averaged mini-batch loss for training
        - use total averaged loss for validation/testing

        TODO:
        1. rearrange the metrics according to tensorboard convention
        2. can we use registration pattern instead of using naming?
        3. can we split this big method into smaller ones? by inherent, hooks 
        4. how to enforce the sub-class to implement the same metrics? Sometimes we forget to implement, or use diff names
        """
        
        # type: bezier curve type
        # L2y: L2 loss of the average position

        #for m in ['ACCh2o', 'LLh2o', 'LLx2cpts']:
        for m in ['rmse']:
            self.loss_names.extend([m+'_'+self.tag])

        return 

    def register_displayImages(self):
        """
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        """
        tag=self.tag
        if self.opt.sw_useImgx:
            self.visual_names.append('img_x_'+tag)
        return 
    
    # ================ input data    



    def set_input(self, input):
        """
            shared method for train/val runners

            arg:
                input: a list of elements from the bezier dataset object directly
                    for picasso01
                    item:= (pimg_tensor, cpts) 
                        cpts:=[(k,2)] a list of sub-strokes, each sub-stroke is kx2 tensor for the control points 

            note:
                the bezier datasets are diff to other datasets, because we use plain collate_fn() in dataloader construction.

            assumptions:
                1. the batch size should be 1, for a single process. unless we load in batch and run in parallel.
                2. sw_useImgx should be True
        """
        # # add in the extra channel 1, in order to work with previous code for image visualisation and so on
        # # pyBPL gives (105, 105) tensors only
        # self.x = input[0]['pimg_tensor'].to(self.model.device).view(1, 1, self.opt.image_hw, self.opt.image_hw)
        # self.x_coord = self.add2dcoord(self.x)

        # # TODO: allow hook and obersers here, and make them automate
        # with torch.no_grad():
        #     if self.opt.sw_useImgx:
        #         setattr(self, 'img_x_'+self.tag, self.x)

        # # move to GPU even for a single tensor, so we can calculate the loss
        # # take the first sub-stroke for picasso01 single stroke model
        # self.cpts=input[0]['cpts'][0].view(1, -1, 2).to(self.model.device) # [batch_size, (cpts, 2)]
        # # the order should be used to select networks
        # self.bezier_order=torch.tensor(self.cpts.shape[1]-1).to(self.model.device) # order starts from 0

        # return 

        # Do not need it for multi-step interactive environment
        pass


    # # ================== sub-network running shared sub-process

    # def get_x2h(self, x):
    #     """
    #         use base convnet
            
    #         use x instead of self.x
    #         incase we would like to repeat the process

    #         but use self.h for what? 
    #         we shoud make a wrapper if we would like to debug
    #         TODO: a debugging wrapper, a function to prob the internal states

    #         output:
    #             h: hidden inter variable
    #     """
    #     h=self.model.netx2h(x)
    #     #h=nnf.relu(h) convnet output is always relued in our network config
    #     h=h.view([1,-1])         # reshape the convnet output

    #     return h
    
    # def get_suploss_h2order(self, h):
    #     """
    #     predict the order of the bezier curve

    #     calculate the loss

    #     the director will sample and choose the corresponding networks for the next step
    #     # TODO: how to make this a standard function? Pass in network and output, get everything we need, do not change the source code anymore
    #     # TODO: how to better prob the internal activation values for debugging? auto-naming them?

    #     # TODO refector: separate loss and run? not useufl if we only do test and train, unless during deploy and semi-sup
    #     """

    #     # MLP output is not relued
    #     logits_h2order=self.model.neth2order(h) # (batch_size, feature len)

    #     # TODO: refactor into the network?
    #     # this is important!?
    #     logits_h2order=torch.nn.functional.log_softmax(logits_h2order)
        
    #     q_h2o = torch.distributions.categorical.Categorical(logits=logits_h2order) # logits are log probs
        
    #     # the self.y input to the log_prob has to be [batch_size] with out extra dim 1. Otherwise, the distribution result is wrong
    #     LL_h2o = q_h2o.log_prob(self.bezier_order.view([1])) # the good thing is that this will work for both binary and multi-class
    #     #LL_h2o = LL_h2o.view([self.cbi, self.cbj])
    #     with torch.no_grad():
    #         # we do not need to care about binary cases here, they are the same. nice
    #         o_hat=torch.argmax(logits_h2order, dim=1, keepdim=True) # this is used for acc calculation, not the sample y_hat
    #         bacc_o_hat=self.classification_acc(self.bezier_order, o_hat) # average acc for this batch
    #         raw_acco=self.raw_acc(self.bezier_order, o_hat) # a vector for each element acc results

    #     # use explicit returned value instead of implicit Runner.attributes assign
    #     # because later we are going to refactor the code for multiple runs
    #     # TODO: how to collect the variables by name automatically? a better analysis of the pattern
    #     r=namedtuple('result', ['LL_h2o','o_hat', 'bacc_o_hat', 'raw_acco'])
    #     return r._make([LL_h2o, o_hat, bacc_o_hat, raw_acco])

    # def get_suploss_x2cpts(self, order, x):
    #     """

    #     supervised loss for the shape control points

    #     for picasso01 model
    #         the whole cpts is a single item list
    #     each item is a kx2 tensor, where k is the order

    #     neth2cptsk predict the kx2 position. each point have 2 positions x and y, which are represented by a gaussian
    #     so the total output should be kx2x2=4k

    #     we need to do reshape to match them

    #     NOTE we use separate conv network to take x as input directly
                
    #     """
    #     # call the corresponding network
    #     degree=order.item()
    #     conv_net=getattr(self.model,"netx2h"+str(degree))
    #     cpts_net=getattr(self.model, "neth2cpts"+str(degree))

    #     h=conv_net(x)
    #     #h=nnf.relu(h) convnet output is always relued in our network config
    #     h=h.view([1,-1])         # reshape the convnet output

    #     dpz=cpts_net(h) # distribution parameters of z

    #     mu, tau = self.model._split_z(dpz)       
    #     mu, tau = self.model.tau_positive(mu, tau)

    #     # we use fixed tau here for supervised learning
    #     # move tensor to device
    #     q_cpts=torch.distributions.normal.Normal(mu,torch.tensor([1.0], device=mu.device)) 
    #     # take mean value for each point
    #     # TODO: stretch the cpts tensor to single dim. no batch dimension?
    #     LL_cpts=q_cpts.log_prob(self.cpts.view([self.opt.batch_size, -1])).mean()

    #     r=namedtuple('result', ['LL_cpts','mu'])
    #     return r._make([LL_cpts, mu])
    

    # def get_suploss_all(self, x):
    #     h=self.get_x2h(x)
    #     r_o = self.get_suploss_h2order(h)
    #     # the order is detached 
    #     r_cpts =self.get_suploss_x2cpts(r_o.o_hat, x)
    #     #return r_cpts+r_o
    #     return concatenate_namedtuples(r_cpts, r_o)
 
    # @staticmethod
    # def classification_acc(y_true, y_hat):
    #     return (y_hat.squeeze().type(torch.uint8) == y_true.squeeze().type(torch.uint8)).type(torch.float).mean()
    
    # @staticmethod
    # def raw_acc(y_true, y_hat):
    #     """
    #         used to debug the acc running-avg
    #     """
    #     return (y_hat.squeeze().type(torch.uint8) == y_true.squeeze().type(torch.uint8))
    

    # def add2dcoord(self, input_tensor):
    #     """
    #     add 2d coordinative value channel
    #     range [0, 1]

    #     see the coordconv paper
    #     and the git repo
    #     https://github.com/walsvid/CoordConv/blob/master/coordconv.py
    #     """        
    #     def init_coord(input_tensor):
    #         batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape

    #         xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
    #         yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

    #         xx_range = torch.arange(dim_y, dtype=torch.int32)
    #         yy_range = torch.arange(dim_x, dtype=torch.int32)
    #         xx_range = xx_range[None, None, :, None]
    #         yy_range = yy_range[None, None, :, None]

    #         xx_channel = torch.matmul(xx_range, xx_ones)
    #         yy_channel = torch.matmul(yy_range, yy_ones)

    #         # transpose y
    #         yy_channel = yy_channel.permute(0, 1, 3, 2)

    #         # 0 to 1
    #         xx_channel = xx_channel.float() / (dim_y - 1)
    #         yy_channel = yy_channel.float() / (dim_x - 1)

    #         # [-1.0, 1.0]
    #         # xx_channel = xx_channel * 2 - 1
    #         # yy_channel = yy_channel * 2 - 1

    #         xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
    #         yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

    #         device = input_tensor.device()
    #         xx_channel = xx_channel.to(device)
    #         yy_channel = yy_channel.to(device)
    #         return xx_channel, yy_channel
        
    #     # init and move to cuda for once only
    #     if not hasattr(self, 'xx_channel'):
    #         self.xx_channel, self.yy_channel = init_coord(input_tensor)

    #     out = torch.cat([input_tensor, self.xx_channel, self.yy_channel], dim=1)

    #     # in case you need r coordinate
    #     # if self.with_r:
    #     #     rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
    #     #     out = torch.cat([out, rr], dim=1)

    #     return out
