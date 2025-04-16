# aim:
# [] clean old code
# [] build flexible decoders
# notes:
#   copied from network_guihua

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as nnf

import numpy as np

###############################################################################
# Helper Functions
# 1. use relu instead of sigmoid for gradient
# 2. smaller network with tanh() for best performance
# 3. bounded Encoder output
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
        else:
            print('not random initialized: %s' % classname)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
        # dataparallel does not work with our settings. Error: "AttributeError: 'DataParallel' object has no attribute 'layers'
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_32':
        net = UnetGenerator(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """
    #def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0): # swap the real/fake label
    def __init__(self, gan_mode, target_real_label=0.0, target_fake_label=1.0): 
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        nc_factor = 4 # originally 8, but reduced for mnist.
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * nc_factor, ngf * nc_factor, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * nc_factor, ngf * nc_factor, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        
        # omited for mnist
        #unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        """Standard forward."""
        return self.net(x)


# updated for VAE
class AttentionEncoder(nn.Module):
    """
    Create an convolutional encoder 
    - [ ] remove potential attention input
    - [ ] fix the hidden dimensions
    - [ ] Relu for the conv, use linear identity instead of sigmoid for the last layer
    """


    def __init__(self, input_nc, latent_dim, num_downs=4, ngf=4, norm_layer=None, use_dropout=False):
        """Construct an conv encoder with attention input
        Parameters:
            input_nc (int)  -- the number of channels in input images
            num_downs (int) -- the number of downsamplings. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer, the outermost
            norm_layer      -- normalization layer

        Output feature maybe a list of tensors in each layer instead of only the last layer tensor.
        
        unfinished: 
        ~~fixed for mnist+letter 64*64~~
        fixed for 32*32
        fixed with bias
        fixed channel nc_factors for compression
        """
        super(AttentionEncoder, self).__init__()
    
        nc_factors = [4, 4, 8, 16] # guihua: set ngf 4, then nc 8, 16, 32, 32. todo: move to parameters.
        #nc_factors = [2, 4, 8, 8] # guihua: set ngf 4, then nc 8, 16, 32, 32. todo: move to parameters.
        #nc_factors = [2, 4, 8, 8] # guihua: set ngf 4, then nc 8, 16, 32, 32. todo: move to parameters.
        #nc_factors = [1, 2, 8, 32, 16]
        #nc_factors = [1, 2, 4, 32]
        #nc_factors = [1, 4, 6, 64] # increase to get clear AED
        #nc_factors = [1, 4, 8, 32] # increase the size so that get more details when replace tanh() with sigmoid()

        #mayflower{
        #assert ngf==4, "first layer channel should be fixed as 4"
        assert num_downs<=4, "no more than 4 layers. we replace relu with tanh/sigmoid in 4th layer"
        # mayflower: we keep the tanh setting, because we have to make sure that the embedding input are comparable withthe label input scale
        #mayflower}

        self.layers = []

        #if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
        #    use_bias = norm_layer.func == nn.InstanceNorm2d
        #else:
        #    use_bias = norm_layer == nn.InstanceNorm2d
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        
        # guihua: inherited bug fix need to multiply by nc_factors[0]
        self.layers.append(AttentionEncoderBlock(input_nc, ngf * nc_factors[0], use_bias, norm_layer, 0))
        self.add_module('encoder_layer_'+str(0), self.layers[0]) # registry, otherwise not included module
        for l in range(1, num_downs):
            l_input_nc  = ngf * nc_factors[min(len(nc_factors)-1, l-1)]
            l_output_nc = ngf * nc_factors[min(len(nc_factors)-1, l  )]    
            self.layers.append(AttentionEncoderBlock(l_input_nc, l_output_nc, use_bias, norm_layer, l))
            self.add_module('encoder_layer_'+str(l), self.layers[l])

        self.layers.append(EncoderLinearBlock(ngf * nc_factors[min(len(nc_factors)-1, l  )], latent_dim))
        self.add_module('encoder_layer_linear', self.layers[-1])

    def forward(self, x, attention_flag=False, channelwise_attention=None):
        """
        output -- a list of activation values and statistics(mean activation) of each layer, in contrast with only last layer activation value. 
        To get the last layer activation as the normal output, use output[-1][0]
        """
        output = []
        for l in range(len(self.layers)):   
            if attention_flag and not channelwise_attention==None:
                l_attention = channelwise_attention[l]
            else:
                l_attention = None
            x, avg_act = self.layers[l](x, attention_flag, l_attention, attention_scaler=None)
            output.append((x, avg_act)) 
        return output

class EncoderLinearBlock(nn.Module):
    """ linear latent layer for VAE, flattern the conv output and then add linear layer"""
    def __init__(self, input_dim, output_dim):
        super(EncoderLinearBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x, attention_flag=False, channelwise_attention=None, attention_scaler=None):
        x = flattern_tensor(x) # convert to 1D
        x = self.fc(x)
        avg_act = x.mean(-1).mean(-1)
        return x, avg_act


class AttentionEncoderBlock(nn.Module):
    """define the attention encoder block"""

    def __init__(self, input_nc, output_nc, use_bias, norm_layer, l, padding=1):
        """
        l: number of layer
        0 the outer most
        -1 the inner most
        """
        super(AttentionEncoderBlock, self).__init__()
        
        self.l = l
        assert l <= 3, "max 4 layers, we change the relu to tahn in the last layer"

        if l < 3 :
            self.non_linear = nn.LeakyReLU(0.2, inplace=True)
            #self.non_linear = nn.Sigmoid()
        else:
            #self.non_linear = nn.Tanh()
            #self.non_linear = nn.Sigmoid() # mayflower: change back from tanh() to sigmoid() because we are going to use this code together with labels.
            
            #self.non_linear = nn.Identity() # for vae: last layer is linear, but not Identity(), it is Relu
            self.non_linear = nn.LeakyReLU(0.2, inplace=True)
        
            padding = 0
        # only last layer padding 0, others padding 1 to scale down by 2 with 4x4 stride 2 kernels.
        self.conv = nn.Conv2d(input_nc, output_nc, kernel_size=4,
                stride=2, padding=padding, bias=use_bias)
        if self.l != 0 and self.l != 3: # not the last layer in the 4-layer encoder 
            self.norm_layer = norm_layer(output_nc) 
    
    def expand_attention(self, x, a, scaler):
        """
        for each channel, attention value are expand from a scalar to a matrix
        same size as the 2d feature map.
        # [batch, channel, width, height]
        # [batch, channel]
        """
        a = a * scaler
        # a = a + 1
        a = a.unsqueeze(-1).expand(a.size(0), a.size(1), x.size(2))
        a = a.unsqueeze(-1).expand_as(x)
        x = x * a
        return x
    
    def forward(self, x, attention_flag=False, channelwise_attention=None, attention_scaler=None):
        
        x = self.conv(x)
        if attention_flag:
            x = self.expand_attention(x, channelwise_attention, attention_scaler)
        
        if self.l != 0 and self.l != 3: # not the last layer in the 4-layer encoder 
            x = self.norm_layer(x) # unfinished: this hurt the 1x1 conv
            
        x = self.non_linear(x)
        #get average reaction, [batch, channel]
        #may also use max or min or any other ones.
        avg_act = x.mean(-1).mean(-1)

        return x, avg_act

class DecoderLinearBlock(nn.Module):
    """ linear latent output layer for VAE generator, 
        1. change the latent dims to be aligned with covolutional features
        2. reshape the output dimensions
    """
    def __init__(self, input_dim, output_nc, output_w, output_h):
        super(DecoderLinearBlock, self).__init__()
        output_dim = output_nc * output_w * output_h
        self.fc = nn.Linear(input_dim, output_dim)
        self.output_nc = output_nc
        self.output_w = output_w
        self.output_h = output_h
    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(x.shape[0], self.output_nc, self.output_w, self.output_h)
        return x

class Decoder(nn.Module):
    """Create an convolutional decoder"""

    def __init__(self, output_nc, latent_dim, input_w, input_h, num_downs=4, ngf=64, norm_layer=None, use_dropout=False):
        """Construct an conv encoder with attention input
        Parameters:
            input_nc (int)  -- the number of channels in input images
            num_downs (int) -- the number of downsamplings. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer, the outermost
            norm_layer      -- normalization layer

        Output feature maybe a list of tensors in each layer instead of only the last layer tensor.
        
        unfinished: fixed for mnist+letter 64*64
        """
        super(Decoder, self).__init__()
        
        nc_factors = [4, 4, 8, 16] # guihua:  todo: aligned with encoder? but the input layer should be halved
        #nc_factors = [2, 4, 4, 4] # guihua:  todo: aligned with encoder? but the input layer should be halved
        #nc_factors = [2, 4, 8, 8] # guihua:  todo: aligned with encoder? but the input layer should be halved
        #nc_factors = [1, 2, 8, 32, 16]
        #nc_factors = [1, 2, 4, 32]
        #nc_factors = [1, 4, 6, 64] # increase to get clear AED
        #nc_factors = [1, 4, 8, 32]
        #nc_factors = [1, 2, 4, 64]
        self.layers = []
        self.layers.append(DecoderLinearBlock(latent_dim, ngf * nc_factors[min(len(nc_factors)-1, num_downs-1)], input_w, input_h))
        self.add_module('decoder_layer_linearinput', self.layers[0])
        #if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
        #    use_bias = norm_layer.func == nn.InstanceNorm2d
        #else:
        #    use_bias = norm_layer == nn.InstanceNorm2d
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        # we name the layers in decoder reversely, so that pretrain enc-dec from outside-in and preload
        # if padding=1, w_output = w_input*2; if padding=0, w_output = w_input*2+2. totally reverse of the convolution padding. we need padding for the inner most layer, othewise, although the kernel size is 4, output is 2x2
        for l in range(0, num_downs-1):
            l_inverse = num_downs-1-l
            l_input_nc  = ngf * nc_factors[min(len(nc_factors)-1, l_inverse)]
            l_output_nc = ngf * nc_factors[min(len(nc_factors)-1, l_inverse-1)]    
            #self.layers.append(DecoderBlock(l_input_nc, l_output_nc, use_bias, norm_layer, l_inverse))
            if num_downs < 4 or l>0:
                l_padding = 1
            else:
                l_padding = 0
            self.layers.append(DecoderBlock(l_input_nc, l_output_nc, use_bias, norm_layer, l_inverse, padding=l_padding, total_l=num_downs))# add padding for 32 example 
            self.add_module('decoder_layer_'+str(l_inverse), self.layers[l+1])
        # guihua: inherited bug, need to multiply the ngf   
        self.layers.append(DecoderBlock(ngf*nc_factors[0], output_nc, use_bias=True, norm_layer=lambda x: Identity(), l=0, padding=1)) # output layer, extra padding for the 48x48 images
        self.add_module('decoder_layer_'+str(0), self.layers[num_downs-1+1])

    def forward(self, x):
        for l in range(len(self.layers)):    
            x = self.layers[l](x)
        return x

class DecoderBlock(nn.Module):
    """define the decoder block"""

    def __init__(self, input_nc, output_nc, use_bias, norm_layer, l, padding=0, total_l=0):
        """
        total_l: added for VAE to make sure that the first layer is linear
        """
        super(DecoderBlock, self).__init__()
        
        self.conv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4,
                stride=2, padding=padding, bias=use_bias)
        self.norm_layer = norm_layer(output_nc)
        self.l = l
        
        # this will be printed as network arch in the log
        #if self.l != total_l-1 and self.l != 0: 
        if self.l != 0: 
            self.non_linear = nn.LeakyReLU(0.2, inplace=True)
        elif self.l == 0:
            self.norm_layer = nn.Identity()
            self.non_linear = nn.Sigmoid()
        #else:
        #    # for the first layer in the VAE decoder
        #    self.non_linear = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        
        x = self.norm_layer(x)
        x = self.non_linear(x) 
        # if self.l != 0:
        #     x = self.norm_layer(x)
        #     #x = nnf.sigmoid(x)
        #     #x = nnf.relu(x)
        #     x = self.non_linear(x)
        # else:
        #     # outermost layer
        #     # x = nnf.tanh(x)
        #     x = nnf.sigmoid(x) # output the parameter of the bernoulli parameter, together with the BCE loss

        return x


def define_Encoder(input_nc, latent_dim, ngf, num_downs=4, netEnc=None, norm='none', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], pretrained=None):
    """Create an encoder

    Parameters:
        input_nc (int) -- the number of channels in input images
        ngf (int) -- the number of filters in the last conv layer
        netEnc (str) -- the architecture's name: 
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        pretrained -- location of pretrained weights
    Returns an encoder

    The encoder has been initialized by <init_net>. It uses leaky RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netEnc == None:
        net = AttentionEncoder(input_nc, latent_dim, num_downs=num_downs, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netEnc)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_Decoder(output_nc, latent_dim, input_w, input_h, ngf, num_downs=4, netDec=None, norm='none', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], pretrained=None):
    """Create an decoder

    unfinished:
    shall we merge enc-dec define?

    Parameters:
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer, input code
        netEnc (str) -- the architecture's name: 
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        pretrained -- location of pretrained weights
    Returns an encoder

    The encoder has been initialized by <init_net>. It uses leaky RELU and tanh for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netDec == None:
        net = Decoder(output_nc, latent_dim, input_w, input_h, num_downs=num_downs, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netDec)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_FbDecoder(output_nc, input_order_nc, ngf, num_downs=4, netDec=None, norm='none', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], pretrained=None):
    """Create an decoder

    unfinished:
    shall we merge enc-dec define?

    Parameters:
        output_nc (int) -- the number of channels in output images
        input_order_nc (int) -- one-hot vector length of the extra topdown order
        ngf (int) -- the number of filters in the last conv layer, input code
        netEnc (str) -- the architecture's name: 
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        pretrained -- location of pretrained weights
    Returns an encoder

    The encoder has been initialized by <init_net>. It uses leaky RELU and tanh for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netDec == None:
        net = FbDecoder(input_order_nc, output_nc, num_downs=num_downs, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netDec)
    return init_net(net, init_type, init_gain, gpu_ids)


class FbDecoder(nn.Module):
    """Create a feedback convolutional decoder"""

    def __init__(self, input_order_nc, output_nc, num_downs=4, ngf=64, norm_layer=None, use_dropout=False):
        """Construct an conv encoder with attention input
        Parameters:
            input_order_nc  -- the number of classes for the order  
            input_nc (int)  -- the number of channels in input images
            num_downs (int) -- the number of downsamplings. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer, the outermost
            norm_layer      -- normalization layer

        Output feature maybe a list of tensors in each layer instead of only the last layer tensor.
        
        We try to give consistent name of layers here, make it easier to reload trained decoder

        unfinished: fixed for mnist+letter 64*64

        """
        super(FbDecoder, self).__init__()
    
        nc_factors = [1, 2, 8, 64, 16]
        self.layers = []

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # we name the layers in decoder reversely, so that pretrain enc-dec from outside-in and preload
        for l in range(0, num_downs-1):
            l_inverse = num_downs-1-l
            l_input_nc  = ngf * nc_factors[min(len(nc_factors)-1, l_inverse)]
            l_output_nc = ngf * nc_factors[min(len(nc_factors)-1, l_inverse-1)]    
            if l == 0: # top-down layer 0
                layer_module = FbDecoderTopBlock(l_input_nc, input_order_nc, l_output_nc, use_bias, norm_layer)
                layer_name = 'fbdecoder_layer_'+str(l_inverse)
            else:
                layer_module = DecoderBlock(l_input_nc, l_output_nc, use_bias, norm_layer)
                layer_name = 'decoder_layer_'+str(l_inverse)
            self.layers.append(layer_module)
            self.add_module(layer_name, self.layers[l])
        
        # output layer, last one
        self.layers.append(DecoderBlock(ngf, output_nc, use_bias=True, norm_layer=lambda x: Identity()))
        self.add_module('decoder_layer_'+str(0), self.layers[num_downs-1])

    def forward(self, x, topdown_c):
        for l in range(len(self.layers)):
            if l == 0 :
                x = self.layers[l](x, topdown_c)
            else:
                x = self.layers[l](x)
            # use tanh for last layer output
            if l == len(self.layers)-1:
                x = nnf.tanh(x)
            else:
                x = nnf.relu(x) 
        return x

class FbDecoderTopBlock(nn.Module):
    """
    used as at top block to update the features according to the input topdown order
    
    """
    def __init__(self, input_nc, input_order_nc, output_nc, use_bias, norm_layer):
        """
        1. extend order
        2. reshape order
        3. concate original input and extended order
        4. conv and output

        unfinished:
        1. fixed features size here. can we resize it?
        """
        super(FbDecoderTopBlock, self).__init__()

        self.fc_back = nn.Linear(input_order_nc, input_nc*4*4) # fixed feature map size here
        #self.conv = nn.ConvTranspose2d(2*input_nc, output_nc, kernel_size=4,
        #        stride=2, padding=1, bias=use_bias) # double input chennel 

        self.conv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4,
                stride=2, padding=1, bias=use_bias) # double input chennel 
        #self.norm_layer = norm_layer(output_nc)

    def forward(self, x, topdown_c):
        """
        topdown_c -- one-hot vector for topdown class signals
        unfinished: fixed the shape here
        """
        a = self.fc_back(topdown_c)
        # shall we use softmax here?
        a = nnf.sigmoid(a)
        #a = nnf.tanh(a)
        #a = nnf.softmax(a, dim=1) # do not softmax beccause that picks single feature
        a = nnf.normalize(a)*100 #d2 norm # do not norm because if input is wrong, there should be no reaction? No
        a = a.reshape((1, 128, 4, 4))
        #x = self.norm_layer(x)
        #x = torch.cat((a, x), 1)
        x = torch.mul(a, x)
        #x = self.norm_layer(x)
        x = self.conv(x)
        return x 


############ Guihua rotation function CNN Translator 
class Rotator(nn.Module):
    """
    CNN rotator
    clearly not the best network structure, just for comparison
    """


    def __init__(self):
        """Construct an conv encoder with attention input
        Parameters:
            input_nc (int)  -- the number of channels in input images
            num_downs (int) -- the number of downsamplings. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer, the outermost
            norm_layer      -- normalization layer

        """
        super(Rotator, self).__init__()
        
        # we fix the network structure here
        input_nc = 1 + 1 # allow rotation as another channel input
        ngf = 1
        num_downs= 4
        #nc_factors = [128, 128, 128, 128, 128]
        nc_factors = [64, 64, 64, 64, 64]
        use_bias = True
        norm_layer = None

        self.layers = []
        self.layers.append(RotatorBlock(input_nc, ngf * nc_factors[0], use_bias))
        self.add_module('rotator_layer_'+str(0), self.layers[0]) # registry, otherwise not included module
        for l in range(1, num_downs):
            l_input_nc  = ngf * nc_factors[min(len(nc_factors)-1, l-1)]
            l_output_nc = ngf * nc_factors[min(len(nc_factors)-1, l  )]    
            self.layers.append(RotatorBlock(l_input_nc, l_output_nc, True))
            self.add_module('rotator_layer_'+str(l), self.layers[l])
        # extra output layer
        self.layers.append(RotatorBlock(ngf * nc_factors[min(len(nc_factors)-1, l  )], 1, True, is_end=True))
        self.add_module('rotator_layer', self.layers[-1])

    def forward(self, x, attention_flag=False, channelwise_attention=None):
        """
        """
        #x = torch.nn.functional.interpolate(x, (16, 16), mode='bicubic')
        x = torch.nn.functional.interpolate(x, (16, 16), mode='bilinear')
        for l in range(len(self.layers)):   
            x= self.layers[l](x)
            #if l == len(self.layers)-2: #second last layer
                #x = torch.nn.functional.interpolate(x, (32, 32), mode='bicubic')
                #x = torch.nn.functional.interpolate(x, (32, 32), mode='bilinear')
        x = torch.nn.functional.interpolate(x, (32, 32), mode='bilinear')
        return x

class RotatorBlock(nn.Module):

    def __init__(self, input_nc, output_nc, use_bias, is_end=False):
        """
        l: number of layer
        0 the outer most
        -1 the inner most
        """
        super(RotatorBlock, self).__init__()
        
        if not is_end:
            self.non_linear = nn.LeakyReLU(0.2, inplace=True)
        else:
            #self.non_linear = nn.Sigmoid()
            self.non_linear = nn.Tanh() # [-1, 1]

        # use not so big kernel, to avoid global complexity
        # use same padding, so that the image is the same size
        padding = 2
        self.conv = nn.Conv2d(input_nc, output_nc, kernel_size=5,
                stride=1, padding=padding, bias=use_bias)
  
    def forward(self, x):
        
        x = self.conv(x)
        x = self.non_linear(x)

        return x

def define_Rotator(init_type='normal', init_gain=0.02, gpu_ids=[], pretrained=None):
    net = None

    if True:
        net = Rotator()
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)

class AERotator(nn.Module):
    """
    CNN rotator
    pure CNN does not work, need to reduce dimension
    clearly not the best network structure, just for comparison
    """


    def __init__(self):
        """Construct an conv encoder with attention input
        Parameters:
            input_nc (int)  -- the number of channels in input images
            num_downs (int) -- the number of downsamplings. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer, the outermost
            norm_layer      -- normalization layer

        """
        super(AERotator, self).__init__()
        
        # we fix the network structure here
        input_nc = 1 # allow rotation as another channel input
        output_nc = 1
        ngf = 1
        #nc_factors = [128, 128, 128, 128, 128]
        
        #nc_factors = [16, 32, 64, 64]
        #enc_padding = [1, 1, 1, 0]
        
        nc_factors = [32, 64, 128]
        enc_padding = [1, 1, 1]
        
        num_downs= len(nc_factors)
        use_bias = True
        norm_layer = None

        self.layers = []

        self.layers.append(AERotatorEncBlock(input_nc, ngf * nc_factors[0], use_bias, padding=enc_padding[0]))
        self.add_module('rotator_layer_'+str(0), self.layers[0]) # registry, otherwise not included module
        for l in range(1, num_downs):
            l_input_nc  = ngf * nc_factors[min(len(nc_factors)-1, l-1)]
            l_output_nc = ngf * nc_factors[min(len(nc_factors)-1, l  )]    
            self.layers.append(AERotatorEncBlock(l_input_nc, l_output_nc, True, padding=enc_padding[l]))
            self.add_module('rotator_layer_'+str(l), self.layers[l])

        #nc_factors = [16, 32, 64, 64+1] # add in last layer
        #nc_factors = [16, 32, 64+1] # add in last layer
        nc_factors = [64, 128, 128+1] # add in last layer
        # we name the layers in decoder reversely, so that pretrain enc-dec from outside-in and preload
        # if padding=1, w_output = w_input*2; if padding=0, w_output = w_input*2+2. totally reverse of the convolution padding. we need padding for the inner most layer, othewise, although the kernel size is 4, output is 2x2
        for l in range(0, num_downs-1):
            l_inverse = num_downs-1-l
            l_input_nc  = ngf * nc_factors[min(len(nc_factors)-1, l_inverse)]
            l_output_nc = ngf * nc_factors[min(len(nc_factors)-1, l_inverse-1)]    
            #if num_downs < 4 or l>0:
            #    l_padding = 1
            #else:
            #    l_padding = 0
            l_padding = enc_padding[min(len(nc_factors)-1, l_inverse)]
            self.layers.append(AERotatorDecBlock(l_input_nc, l_output_nc, use_bias, norm_layer, l_inverse, padding=l_padding, total_l=num_downs))# add padding for 32 example 
            self.add_module('decoder_layer_'+str(l_inverse), self.layers[-1])

        self.layers.append(AERotatorDecBlock(ngf*nc_factors[0], output_nc, use_bias=True, norm_layer=lambda x: Identity(), l=0, padding=1)) # output layer, extra padding for the 48x48 images
        self.add_module('decoder_layer_'+str(0), self.layers[-1])

    def forward(self, x, r, attention_flag=False, channelwise_attention=None):
        """
        """
        #x = torch.nn.functional.interpolate(x, (16, 16), mode='bilinear')
        for l in range(len(self.layers)):  
            if l == 3:
                x = torch.cat((x,r), 1) # add the rotation
            x= self.layers[l](x)
        #x = torch.nn.functional.interpolate(x, (32, 32), mode='bilinear')
        return x

class AERotatorEncBlock(nn.Module):

    def __init__(self, input_nc, output_nc, use_bias, padding=1, is_end=False):
        """
        l: number of layer
        0 the outer most
        -1 the inner most
        """
        super(AERotatorEncBlock, self).__init__()
        
        if not is_end:
            self.non_linear = nn.LeakyReLU(0.2, inplace=True)
        else:
            #self.non_linear = nn.Sigmoid()
            self.non_linear = nn.Tanh() # [-1, 1]

        # use not so big kernel, to avoid global complexity
        self.conv = nn.Conv2d(input_nc, output_nc, kernel_size=4,
                stride=2, padding=padding, bias=use_bias)
  
    def forward(self, x):
        
        x = self.conv(x)
        x = self.non_linear(x)

        return x

class AERotatorDecBlock(nn.Module):
    """define the decoder block"""

    def __init__(self, input_nc, output_nc, use_bias, norm_layer, l, padding=0, total_l=0):
        """
        total_l: added for VAE to make sure that the first layer is linear
        """
        super(AERotatorDecBlock, self).__init__()
        
        self.conv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4,
                stride=2, padding=padding, bias=use_bias)
        self.l = l
        
        # this will be printed as network arch in the log
        #if self.l != total_l-1 and self.l != 0: 
        if self.l != 0: 
            self.non_linear = nn.LeakyReLU(0.2, inplace=True)
        elif self.l == 0:
            #self.non_linear = nn.Sigmoid()
            self.non_linear = nn.Tanh()
    def forward(self, x):
        x = self.conv(x)
        x = self.non_linear(x) 
        return x
def define_AERotator(init_type='normal', init_gain=0.02, gpu_ids=[], pretrained=None):
    net = None

    if True:
        net = AERotator()
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)

######### TransCEncDec

def define_CTranslator(input_t_length, input_c_length, output_c_length, init_type='normal', init_gain=0.02, gpu_ids=[], pretrained=None):
    """Create a translator
    
    arguments:
    input_t_length -- t is topdown input
    input_c_length -- original code
    output_c_length -- new code

    Returns a translator

    The encoder has been initialized by <init_net>. It uses leaky RELU and tanh for non-linearity.
    """
    net = None

    if True:
        net = CTranslator(input_t_length, input_c_length, output_c_length)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


class CTranslator(nn.Module):
    """Create a MLU tanslator"""

    def __init__(self, input_t_length, input_c_length, output_c_length):
        """Construct MLU
       
        arguments:
        input_t_length -- t is topdown input
        input_c_length -- original code
        output_c_length -- new code
        
        unfinished: fixed for 2 layers here

        """
        super(CTranslator, self).__init__()
        
        input_length = input_t_length+input_c_length
        self.fc1 = nn.Linear(input_length, input_length*2)
        self.fc15 = nn.Linear(input_length*2, input_length*2)
        self.fc2 = nn.Linear(input_length*2, output_c_length)
        
    def forward(self, t, c):
        #deprecated: # rescale x so that it is comparable or smaller than the code 1.0 in value
        # c = c*0.01
        #h = h.to(self.device)
        if len(t.size()) > 2:
            flatten_t = t.reshape(1, -1)
        else:
            flatten_t = t
        flatten_c = c.reshape(1, -1)
        x = torch.cat((flatten_c, flatten_t), 1)
        x = self.fc1(x)
        
        #x = nnf.relu(x) 
        
        #x = torch.cat((x, t), 1)
        #x = self.fc15(x)
        x = nnf.tanh(x) # use tanh() here to be comparable with the original output code
        
        #x = torch.cat((x, t), 1)
        x = self.fc2(x)
        # no relu here, because the input x is not relued
        #x = nnf.relu(x)
        #x = x.reshape(1, 64, 4, 4)
        x = x.reshape(1, 128, 1, 1)
        
        # x = x*100.0
        return x

########## CTemplator 

def define_CTemplator(input_h_length, output_t_length, init_type='normal', init_gain=0.02, gpu_ids=[], pretrained=None):
    """Create a Templator
    
    arguments:
    input_h_length -- h is topdown input, one hot
    output_c_length -- output template code

    Returns a Templator

    """
    net = None

    if True:
        net = CTemplator(input_h_length, output_t_length)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


class CTemplator(nn.Module):
    """Create a templator, a memory actually
    use a linear translation only
    """

    def __init__(self, input_h_length, output_t_length):
        """Construct MLU
       
        arguments:
        input_h_length -- h is topdown input
        output_c_length -- output template code

        unfinished:
        1. can we have different learning rate here?
        2. test use sigmoid? not really necessary
        """
        super(CTemplator, self).__init__()
        
        self.memorymatrix = nn.Linear(input_h_length, output_t_length)
        
    def forward(self, h):
        """
        retrive the memory
        """
        template = self.memorymatrix(h) 
        return template

######## CBlender

def define_CBlender(input_t_length, input_c_length, output_c_length, init_type='normal', init_gain=0.02, gpu_ids=[], pretrained=None):
    """
    Instead of translating, blender takes in both top-down and bottom-up and current-code, and optmize the current-code
    
    arguments:
    input_t_length -- t is topdown input
    input_c_length -- original code
    output_c_length -- new code

    Returns a translator

    The encoder has been initialized by <init_net>. It uses leaky RELU and tanh for non-linearity.
    """
    net = None

    if True:
        net = CBlender(input_t_length, input_c_length, output_c_length)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


class CBlender(nn.Module):
    """Create a MLU tanslator"""

    def __init__(self, input_t_length, input_c_length, output_c_length):
        """Construct MLU
       
        arguments:
        input_t_length -- t is topdown input
        input_c_length -- original code
        output_c_length -- new code
        
        unfinished: fixed for 2 layers here

        """
        super(CBlender, self).__init__()

        #assert input_t_length == input_c_length, "top-down and bottom-up code length should be the same"
        assert input_c_length == output_c_length, "output code is only a changed version of input"
        
        input_length = input_t_length + input_c_length
        self.fc1 = nn.Linear(input_length, input_length*4)
        self.fc15 = nn.Linear(input_length*4, input_length*4)
        self.fc2 = nn.Linear(input_length*4, output_c_length)
        
    def forward(self, c, t):
        """
        t: top-down hypo
        c: bottom-up code
        do not use template to avoid overfitting
        """
        if len(t.size()) > 2:
            flatten_t = t.reshape(1, -1)
        else:
            flatten_t = t

        flatten_c = c.reshape(1, -1)

        x = torch.cat((flatten_c, flatten_t), 1)
        x = self.fc1(x)
        x = nnf.relu(x) 
        
        x = self.fc15(x)
        x = nnf.relu(x)
        
        x = self.fc2(x)
        x = nnf.tanh(x) # use tanh() here to be comparable with the original output code
        
        x = x.reshape(1, 128, 1, 1)
        return x

######## Spliter

def define_Spliter(z_length, width_factor=4, init_type='normal', init_gain=0.02, gpu_ids=[], pretrained=None):
    """
    # spliter
    split directly from z0 and z1 to z2

    # blender
    Instead of translating, blender takes in both top-down and bottom-up and current-code, and optmize the current-code
    
    arguments:
    input_t_length -- t is topdown input
    input_c_length -- original code
    output_c_length -- new code

    Returns a translator

    The encoder has been initialized by <init_net>. It uses leaky RELU and tanh for non-linearity.
    """
    net = None

    if True:
        net = Spliter(z_length, width_factor=4)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


class Spliter(nn.Module):
    """Create a MLU spliter"""

    def __init__(self, z_length, width_factor):
        """Construct MLU
       
        arguments:
        - z_length -- assume input 2 z_length, output 1 z_length vector
        
        decisions: 
        - fixed for 2 layers here
        """
        super(Spliter, self).__init__()
        
        assert z_length==128, "currently fix design for 128"
        self.z_length = z_length
        
        input_length = 2*z_length
        output_length = z_length
        self.fc1 = nn.Linear(input_length, input_length*width_factor)
        self.fc15 = nn.Linear(input_length*width_factor, input_length*width_factor)
        self.fc2 = nn.Linear(input_length*width_factor, output_length)
        
    def forward(self, z0, z1):
        """
        z0 is the mixed code
        z1 is one of the splited code
        """
        if len(z1.size()) > 2:
            flatten_z1 = z1.reshape(1, -1)
        else:
            flatten_z1 = z1

        flatten_z0 = z0.reshape(1, -1)

        x = torch.cat((flatten_z0, flatten_z1), 1)
        x = self.fc1(x)
        x = nnf.relu(x) 
        
        x = self.fc15(x)
        x = nnf.relu(x)
        
        x = self.fc2(x)
        x = nnf.tanh(x) # use tanh() here to be comparable with the original output code
        
        x = x.reshape(1, self.z_length, 1, 1)
        return x


####### GAN
def flattern_tensor(x):
    """
    flattern the tensor
    Dimension of x: [batch_size, channel, d1, d2]
    """
    batch_size = x.shape[0]

    if len(x.size()) > 2:
        flatten = x.reshape(batch_size, -1)
    else:
        flatten = x
    return flatten


class MLPDiscriminator(nn.Module):
    """Defines a MLP discriminator"""

    def __init__(self, input_length, width_factor):
        """Construct a discriminator

        """
        super(MLPDiscriminator, self).__init__()
       
        self.fc1 = nn.Linear(input_length, input_length*width_factor)
        self.fc2 = nn.Linear(input_length*width_factor, input_length*width_factor)
        self.fc3 = nn.Linear(input_length*width_factor, 1)
        
        self.relu = nn.LeakyReLU(0.2)
    def forward(self, inputs):
        """
        inputs: a list of tensor, total length should be the same as initialised
        """
        
        
        inputs = [flattern_tensor(x) for x in inputs]
        if len(inputs) >1:
            x = torch.cat(inputs, 1)
        else:
            x = inputs[0]

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        #x = nnf.sigmoid(x) # do not use this sigmoid. we deal with bcewithlogitscloss

        return x

def define_MLPD(input_length, width_factor=4, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = MLPDiscriminator(input_length, width_factor) 
    # raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)

######## SpliterAll

def define_SpliterAll(z_length, init_type='normal', init_gain=0.02, gpu_ids=[], pretrained=None):
    """
    # spliterall
    take both z1 and z2 as input, output

    # spliter
    split directly from z0 and z1 to z2

    # blender
    Instead of translating, blender takes in both top-down and bottom-up and current-code, and optmize the current-code
    
    arguments:
    input_t_length -- t is topdown input
    input_c_length -- original code
    output_c_length -- new code

    Returns a translator

    The encoder has been initialized by <init_net>. It uses leaky RELU and tanh for non-linearity.
    """
    net = None

    if True:
        net = SpliterAll(z_length)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


class SpliterAll(nn.Module):
    """Create a MLU spliter"""

    def __init__(self, z_length):
        """Construct MLU
       
        arguments:
        - z_length -- assume input 2 z_length, output 1 z_length vector
        
        decisions: 
        - fixed for 2 layers here
        """
        super(SpliterAll, self).__init__()
        
        assert z_length==128, "currently fix design for 128"
        self.z_length = z_length
        
        input_length = 3*z_length
        output_length = 2*z_length
        self.fc1 = nn.Linear(input_length, input_length*4)
        self.fc15 = nn.Linear(input_length*4, input_length*4)
        self.fc16 = nn.Linear(input_length*4, input_length*4)
        self.fc2 = nn.Linear(input_length*4, output_length)
        
        self.relu = nn.LeakyReLU(0.2)
    def forward(self, z0, z1, z2):
        """
        z0 is the mixed code
        z1 z2, is one of the splited code
        """
        x = torch.cat((flattern_tensor(z0), flattern_tensor(z1), flattern_tensor(z2)), 1)
        x = self.fc1(x)
        x = self.relu(x) 
        
        x = self.fc15(x)
        x = self.relu(x)
        
        x = self.fc16(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = nnf.tanh(x) # use tanh() here to be comparable with the original output code
       
        x1 = x[0, 0:self.z_length]
        x2 = x[0, self.z_length:]
        x1 = x1.reshape(1, self.z_length, 1, 1)
        x2 = x2.reshape(1, self.z_length, 1, 1)
        return x1, x2
######## DAE

def define_DAE(z_length, width_factor=4, init_type='normal', init_gain=0.02, gpu_ids=[], pretrained=None):
    """
    # DAE
    z, z+, z- as input
    z' as output

    # spliterall
    take both z1 and z2 as input, output

    # spliter
    split directly from z0 and z1 to z2

    # blender
    Instead of translating, blender takes in both top-down and bottom-up and current-code, and optmize the current-code
    
    arguments:
    input_t_length -- t is topdown input
    input_c_length -- original code
    output_c_length -- new code

    Returns a translator

    The encoder has been initialized by <init_net>. It uses leaky RELU and tanh for non-linearity.
    """
    net = None

    if True:
        net = DAE(z_length, width_factor=width_factor)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


class DAE(nn.Module):
    """Create a MLU spliter"""

    def __init__(self, z_length, width_factor):
        """Construct MLU
       
        arguments:
        - z_length -- assume input 2 z_length, output 1 z_length vector
        
        decisions: 
        - fixed for 2 layers here
        """
        super(DAE, self).__init__()
        
        assert z_length==128, "currently fix design for 128"
        self.z_length = z_length
        
        input_length = 3*z_length
        output_length = z_length
        self.fc1 = nn.Linear(input_length, input_length*width_factor)
        self.fc15 = nn.Linear(input_length*width_factor, input_length*width_factor)
        #self.fc16 = nn.Linear(input_length*width_factor, input_length*width_factor)
        self.fc2 = nn.Linear(input_length*width_factor, output_length)
        
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, z0, z1, z2):
        """
        z0 is the current estimation
        z1 = z+ the one need to explain
        z2 = z- the one need to avoid
        """
        x = torch.cat((flattern_tensor(z0), flattern_tensor(z1), flattern_tensor(z2)), 1)
        x = self.fc1(x)
        x = self.relu(x) 
        
        x = self.fc15(x)
        x = self.relu(x)
        
        #x = self.fc16(x)
        #x = self.relu(x)
        
        x = self.fc2(x)
        x = nnf.tanh(x) # use tanh() here to be comparable with the original output code
       
        x = x.reshape(1, self.z_length, 1, 1)
        return x

############ MnistClassifier
class MLPClassifier(nn.Module):
    """Defines a MLP classifier"""

    def __init__(self, input_length, output_length, width_factor):
        """Construct a discriminator

        """
        super(MLPClassifier, self).__init__()
       
        self.fc1 = nn.Linear(input_length, output_length*width_factor*2)
        self.fc2 = nn.Linear(output_length*width_factor*2, output_length*width_factor)
        self.fc3 = nn.Linear(output_length*width_factor, output_length)
        
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, inputs):
        """
        inputs: a single tensor, total length should be the same as initialised
        """
        
        x = flattern_tensor(inputs)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        #x = nnf.sigmoid(x) # do not use this sigmoid. we use CrossEntropyLoss in classifier 

        return x

def define_MLPC(input_length, output_length, width_factor=4, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a classifier

    Parameters:
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Our current implementation provides three types of discriminators:
    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = MLPClassifier(input_length, output_length, width_factor) 
    # raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


######## DAE

def define_DAE2(z_length, width_factor=4, init_type='normal', init_gain=0.02, gpu_ids=[], pretrained=None):
    """
    # DAE
    z, z+, z- as input
    z' as output

    # spliterall
    take both z1 and z2 as input, output

    # spliter
    split directly from z0 and z1 to z2

    # blender
    Instead of translating, blender takes in both top-down and bottom-up and current-code, and optmize the current-code
    
    arguments:
    input_t_length -- t is topdown input
    input_c_length -- original code
    output_c_length -- new code

    Returns a translator

    The encoder has been initialized by <init_net>. It uses leaky RELU and tanh for non-linearity.
    """
    net = None

    if True:
        net = DAE2(z_length, width_factor=width_factor)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


class DAE2(nn.Module):
    """Create a MLU spliter"""

    def __init__(self, z_length, width_factor):
        """Construct MLU
       
        arguments:
        - z_length -- assume input 2 z_length, output 1 z_length vector
        
        decisions: 
        - fixed for 2 layers here
        """
        super(DAE2, self).__init__()
        
        assert z_length==128, "currently fix design for 128"
        self.z_length = z_length
        
        input_length = 2*z_length
        output_length = z_length
        self.fc1 = nn.Linear(input_length, input_length*width_factor)
        self.fc15 = nn.Linear(input_length*width_factor, input_length*width_factor)
        #self.fc16 = nn.Linear(input_length*width_factor, input_length*width_factor)
        self.fc2 = nn.Linear(input_length*width_factor, output_length)
        
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, z0, z1):
        """
        z0 is the current estimation
        z1 = z+ the one need to explain
        z2 = z- the one need to avoid
        """
        x = torch.cat((flattern_tensor(z0), flattern_tensor(z1)), 1)
        #x = flattern_tensor(z)
        x = self.fc1(x)
        x = self.relu(x) 
        
        x = self.fc15(x)
        x = self.relu(x)
        
        #x = self.fc16(x)
        #x = self.relu(x)
        
        x = self.fc2(x)
        x = nnf.tanh(x) # use tanh() here to be comparable with the original output code
       
        x = x.reshape(1, self.z_length, 1, 1)
        return x


############ new MLPClassifier

def flattern_batch_tensor(x):
    """
    flattern the tensor, assuming batchsize >= 1
    Dimension of x: [batch_size, channel, d1=1, d2=1]
    output: [batch_size, channel]
    """
    if len(x.size()) > 2:
        batch_size = x.shape[0]
        n_channel = x.shape[1]
        flatten = x.reshape(batch_size, n_channel)
    else:
        flatten = x
    return flatten

class WMLPClassifier(nn.Module):
    """
    Defines a MLP classifier with width parameter

    different with previous MLPClassifier, which fixed the width factor, we only allow finer defination of layer width, through new paramters.
    
    improve:
    1.[ ] make the layer number flexible
    """

    def __init__(self, input_length, output_length, l_width):
        super(WMLPClassifier, self).__init__()
        
        assert len(l_width) == 3 # hardcode the layer number for current code version
        
        self.fc1 = nn.Linear(input_length, l_width[0])
        self.fc2 = nn.Linear(l_width[0], l_width[1])
        self.fc3 = nn.Linear(l_width[1], l_width[2])
        self.fc4 = nn.Linear(l_width[2], output_length)
        
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, inputs):
        """
        inputs: a batch of 1D tensor
        """
        
        # allow batchsize > 1, no need for flatterning here, torch can handle the extra empty deminsions.  #x = flattern_tensor(inputs)
        x = flattern_batch_tensor(inputs)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        x = self.fc4(x)
        #x = nnf.sigmoid(x) # do not use this sigmoid. we use CrossEntropyLoss in classifier 

        return x

def define_WMLPC(input_length, output_length, l_width, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a classifier

    Parameters:
        l_width(int list)  -- the width of each layer. Length is the layer number.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Our current implementation provides three types of discriminators:
    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = WMLPClassifier(input_length, output_length, l_width) 
    # raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)

############
# solent vae-gan
#    - [x] remove attention input, change output content
#    - [x] remove the ngf
#    - [ ] update the model interface, by adding in nc_factors and remove attentions
############

class PlainEncoder(nn.Module):
    """
    Create an convolutional encoder 
    """
    def __init__(self, input_nc, latent_dim, nc_factors, decinput_wh, num_downs=4, norm_layer=None, use_dropout=False, allow_branch=False):
        """Construct an conv encoder with attention input
        Parameters:
            input_nc (int)  -- the number of channels in input images
            num_downs (int) -- the number of downsamplings. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            removed ngf (int)       -- the number of filters in the last conv layer, the outermost
            norm_layer      -- normalization layer

            allow_branch -- whether we want to output the intermedia results

        Output feature maybe a list of tensors in each layer instead of only the last layer tensor.
        
        unfinished: 
        ~~fixed for mnist+letter 64*64~~
        fixed for 32*32
        fixed with bias
        fixed channel nc_factors for compression
        """
        super(PlainEncoder, self).__init__()
        #nc_factors = [4, 4, 8, 16] # guihua: set ngf 4, then nc 8, 16, 32, 32. todo: move to parameters.
        # assert ngf == nc_factors[-1]
        
        self.allow_branch = allow_branch 

        self.layers = []

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        
        # guihua: inherited bug fix need to multiply by nc_factors[0]
        self.layers.append(PlainEncoderBlock(input_nc, nc_factors[0], use_bias, norm_layer, 0))
        self.add_module('encoder_layer_'+str(0), self.layers[0]) # registry, otherwise not included module

        # padding need to be aligned with decoder, especially the last output layer
        for l in range(1, num_downs):
            l_input_nc  = nc_factors[min(len(nc_factors)-1, l-1)]
            l_output_nc = nc_factors[min(len(nc_factors)-1, l  )]    
            #if l == num_downs-1:
            #    #padding = 0 # the final layer is not padded
            #    padding = 1 # the final layer is not padded
            #else:
            #    padding = 1
            padding = 1
            self.layers.append(PlainEncoderBlock(l_input_nc, l_output_nc, use_bias, norm_layer, l, padding=padding))
            self.add_module('encoder_layer_'+str(l), self.layers[l])

        self.layers.append(PlainEncoderLinearBlock(decinput_wh*decinput_wh*nc_factors[min(len(nc_factors)-1, l  )], latent_dim))
        self.add_module('encoder_layer_linear', self.layers[-1])

    def forward(self, x, l0=0, l1=-1):
        """
        forward up to layer l
        """
        if l1 == -1:
            l1 = len(self.layers)

        for l in range(l0, l1):   
            x, raw_x = self.layers[l](x)
        
        if not self.allow_branch:
            return x
        else:
            return x, raw_x

class PlainEncoderLinearBlock(nn.Module):
    """ linear latent layer for VAE, flattern the conv output and then add linear layer"""
    def __init__(self, input_dim, output_dim):
        super(PlainEncoderLinearBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = flattern_tensor(x) # convert to 1D
        x = self.fc(x) # direct output without non-linear function
        return x, x # output two value to align with other layers

class PlainEncoderBlock(nn.Module):
    """define the attention encoder block"""

    def __init__(self, input_nc, output_nc, use_bias, norm_layer, l, padding=1):
        """
        l: number of layer
        0 the outer most
        -1 the inner most
        """
        super(PlainEncoderBlock, self).__init__()
        
        self.l = l
        self.non_linear = nn.LeakyReLU(0.2, inplace=True)
        # only last layer padding 0, others padding 1 to scale down by 2 with 4x4 stride 2 kernels.
        self.conv = nn.Conv2d(input_nc, output_nc, kernel_size=4,
                stride=2, padding=padding, bias=use_bias)
        
        #if self.l != 0 and self.l != 3: # not the last layer in the 4-layer encoder 
        #    self.norm_layer = norm_layer(output_nc) 
        self.norm_layer = norm_layer(output_nc) 
   
    def forward(self, x):
        
        x = self.conv(x)
        
        raw_x = x # we output raw x in branch 

        #if self.l != 0 and self.l != 3: # not the last layer in the 4-layer encoder 
        #    x = self.norm_layer(x) # unfinished: this hurt the 1x1 conv
        x = self.norm_layer(x) # unfinished: this hurt the 1x1 conv
        x = self.non_linear(x)
        return x, raw_x 

class PlainDecoderLinearBlock(nn.Module):
    """ linear latent output layer for VAE generator, 
        1. change the latent dims to be aligned with covolutional features
        2. reshape the output dimensions
    """
    def __init__(self, input_dim, output_nc, output_w, output_h):
        super(PlainDecoderLinearBlock, self).__init__()
        output_dim = output_nc * output_w * output_h
        self.fc = nn.Linear(input_dim, output_dim)
        self.output_nc = output_nc
        self.output_w = output_w
        self.output_h = output_h
    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(x.shape[0], self.output_nc, self.output_w, self.output_h)
        return x, x # return two values to align with other layers for branching

class PlainDecoder(nn.Module):
    """Create an convolutional decoder"""

    def __init__(self, output_nc, latent_dim, input_w, input_h, nc_factors, num_downs=4, norm_layer=None, use_dropout=False, allow_branch=False):
        """Construct an conv encoder with attention input
        Parameters:
            input_nc (int)  -- the number of channels in input images
            num_downs (int) -- the number of downsamplings. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            removed ngf (int)       -- the number of filters in the last conv layer, the outermost
            norm_layer      -- normalization layer

        Output feature maybe a list of tensors in each layer instead of only the last layer tensor.
        """
        super(PlainDecoder, self).__init__()
        self.allow_branch = allow_branch 
        #nc_factors = [4, 4, 8, 16] # guihua:  todo: aligned with encoder? but the input layer should be halved
        
        self.layers = []
        self.layers.append(PlainDecoderLinearBlock(latent_dim, nc_factors[min(len(nc_factors)-1, num_downs-1)], input_w, input_h))
        self.add_module('decoder_layer_linearinput', self.layers[0])
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        # we name the layers in decoder reversely, so that pretrain enc-dec from outside-in and preload
        # if padding=1, w_output = w_input*2; if padding=0, w_output = w_input*2+2. totally reverse of the convolution padding. we need padding for the inner most layer, othewise, although the kernel size is 4, output is 2x2
        for l in range(0, num_downs-1):
            l_inverse = num_downs-1-l
            l_input_nc  = nc_factors[min(len(nc_factors)-1, l_inverse)]
            l_output_nc = nc_factors[min(len(nc_factors)-1, l_inverse-1)]    
            #if num_downs < 4 or l>0:
            #if l>0:
            #    l_padding = 1
            #else:
            #    #l_padding = 0
            #    l_padding = 1
            l_padding = 1
            self.layers.append(PlainDecoderBlock(l_input_nc, l_output_nc, use_bias, norm_layer, l_inverse, padding=l_padding, total_l=num_downs))# add padding for 32 example 
            self.add_module('decoder_layer_'+str(l_inverse), self.layers[l+1])

        # final layer. TODO: simplify the nc_factor code
        self.layers.append(PlainDecoderBlock(nc_factors[0], output_nc, use_bias=True, norm_layer=lambda x: Identity(), l=0, padding=1)) 
        self.add_module('decoder_layer_'+str(0), self.layers[num_downs-1+1])

    def forward(self, x, l0=0, l1=-1):
        """
        forward up to layer l
        """
        if l1 == -1:
            l1 = len(self.layers)

        for l in range(l0, l1):   
            x, raw_x = self.layers[l](x)
        
        if not self.allow_branch:
            return x
        else:
            return x, raw_x

class PlainDecoderBlock(nn.Module):
    """define the decoder block
    
        note the final layers activation function maybe different
    """

    def __init__(self, input_nc, output_nc, use_bias, norm_layer, l, padding=0, total_l=0):
        """
        total_l: added for VAE to make sure that the first layer is linear
        """
        super(PlainDecoderBlock, self).__init__()
       
        self.l = l
        
        # this will be printed as network arch in the log
        if self.l != 0: 
            # Hout=(Hin−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
            # default dilation = 1, output_padding = 0. with padding = 1, Hout = 2*Hin0
            self.conv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4,
                    stride=2, padding=padding, bias=use_bias)
            self.norm_layer = norm_layer(output_nc)
            self.non_linear = nn.LeakyReLU(0.2, inplace=True)
        elif self.l == 0:
            #self.conv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4,
            #        stride=2, padding=padding, bias=use_bias)
            self.conv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=1,
                    stride=1, padding=0, bias=use_bias)
            self.norm_layer = nn.Identity()
            #self.non_linear = nn.Sigmoid()
            self.non_linear = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        raw_x = x    
        x = self.norm_layer(x)
        x = self.non_linear(x) 
        return x, raw_x


def define_PlainEncoder(input_nc, latent_dim, nc_factors, decinput_wh, num_downs=4, netEnc=None, norm='none', use_dropout=False, allow_branch=False, init_type='normal', init_gain=0.02, gpu_ids=[], pretrained=None):
    """Create an encoder

    Parameters:
        input_nc (int) -- the number of channels in input images
        ngf (int) -- the number of filters in the last conv layer
        netEnc (str) -- the architecture's name: 
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        pretrained -- location of pretrained weights
    Returns an encoder

    The encoder has been initialized by <init_net>. It uses leaky RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netEnc == None:
        net = PlainEncoder(input_nc, latent_dim, nc_factors, decinput_wh, num_downs=num_downs, norm_layer=norm_layer, use_dropout=use_dropout, allow_branch=allow_branch)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netEnc)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_PlainDecoder(output_nc, latent_dim, input_w, input_h, nc_factors, num_downs=4, netDec=None, norm='none', use_dropout=False, allow_branch=False, init_type='normal', init_gain=0.02, gpu_ids=[], pretrained=None):
    """Create an decoder

    unfinished:
    shall we merge enc-dec define?

    Parameters:
        input_w input_h required for VAE where the latent need to be reshaped to a CNN

        output_nc (int) -- the number of channels in output images
        removed ngf (int) -- the number of filters in the last conv layer, input code
        netEnc (str) -- the architecture's name: 
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        pretrained -- location of pretrained weights
    Returns an encoder

    The encoder has been initialized by <init_net>. It uses leaky RELU and tanh for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netDec == None:
        net = PlainDecoder(output_nc, latent_dim, input_w, input_h, nc_factors, num_downs=num_downs, norm_layer=norm_layer, use_dropout=use_dropout, allow_branch=allow_branch)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netDec)
    return init_net(net, init_type, init_gain, gpu_ids)

######### for solent05evae

class LinearReshaper(nn.Module):
    """ 
    linear latent output layer for VAE parts generator, 
        1. change the latent dims to be aligned with covolutional features
        2. reshape the output dimensions
    """
    def __init__(self, input_dim, output_nc, output_w, output_h):
        super(LinearReshaper, self).__init__()
        output_dim = output_nc * output_w * output_h
        self.fc = nn.Linear(input_dim, output_dim)
        self.output_nc = output_nc
        self.output_w = output_w
        self.output_h = output_h
    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(x.shape[0], self.output_nc, self.output_w, self.output_h)
        return x


def define_LinearReshaper(input_dim, output_nc, output_w, output_h, init_type='normal', init_gain=0.02, gpu_ids=[], pretrained=None):
    """
    Create an simple linear layer
    
    Parameters:
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    """
    net = LinearReshaper(input_dim, output_nc, output_w, output_h)
    return init_net(net, init_type, init_gain, gpu_ids)


class FCNDecoder(nn.Module):
    """
    a very simple CNN layer network, without linear layers
    fully control on the channel size, and last layer output type
    """

    def __init__(self, nc_factors, norm_layer=None, last_nonlinearFun='ReLU', last_stride=2, last_padding=1, last_kernel=4,  printname='decoder', allow_branch=False):
        """Construct an conv encoder with attention input

        Note!
        We change nc_factors here from reverse the encoder to independent and normal order.
        Including the input channel number and output channel number, it is a FCN

        TODO: change the definition with list of parameters

        Parameters:
           norm_layer      -- normalization layer

        Output feature maybe a list of tensors in each layer instead of only the last layer tensor.
        """
        super(FCNDecoder, self).__init__()
        
        self.layers = []
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        # we name the layers in decoder reversely, so that pretrain enc-dec from outside-in and preload
        # if padding=1, w_output = w_input*2; if padding=0, w_output = w_input*2+2. totally reverse of the convolution padding. we need padding for the inner most layer, othewise, although the kernel size is 4, output is 2x2
        for l in range(0, len(nc_factors)-2): # all but the last layer
            l_input_nc  = nc_factors[l]
            l_output_nc = nc_factors[l+1]    
            self.layers.append(FCNDecoderBlock(l_input_nc, l_output_nc, use_bias, norm_layer,stride=2, padding=1)) 
            self.add_module(printname+'_layer_'+str(l), self.layers[l])

        # final layer.         
        self.layers.append(FCNDecoderBlock(nc_factors[-2], nc_factors[-1], use_bias=True, norm_layer=lambda x: Identity(), stride=last_stride, padding=last_padding, non_linear=last_nonlinearFun, kernel_size=last_kernel)) 
        self.add_module(printname+'_layer_'+str(len(nc_factors)-1), self.layers[-1])

    def forward(self, x):
        for l in range(len(self.layers)):    
            x = self.layers[l](x)
        return x

def define_FCNDecoder(nc_factors, netDec=None, norm='none', last_nonlinearFun='ReLU', last_stride=2, last_padding=1, last_kernel=4,  printname='decoder', allow_branch=False, init_type='normal', init_gain=0.02, gpu_ids=[], pretrained=None):

    """Create an decoder

    NOTE! the default settings are used by the models. So be careful if you want to change them.

    unfinished:
    shall we merge enc-dec define?
    
    Parameters:
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        pretrained -- location of pretrained weights
    Returns an encoder

    The encoder has been initialized by <init_net>. It uses leaky RELU and tanh for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netDec == None:
        net = FCNDecoder(nc_factors, norm_layer=norm_layer, last_nonlinearFun=last_nonlinearFun, last_stride=last_stride, last_padding=last_padding, last_kernel=last_kernel, printname=printname, allow_branch=allow_branch)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netDec)
    return init_net(net, init_type, init_gain, gpu_ids)

class FCNDecoderBlock(nn.Module):
    """
    define the decoder block
    
    we changed the original design, to allow better control on the non-linear function

    remove the l indexes, give explicit parameters over the layer structure: stride, non-linear
    """

    def __init__(self, input_nc, output_nc, use_bias, norm_layer, non_linear='LeakyReLU', stride=2, padding=1, kernel_size=4):
        super(FCNDecoderBlock, self).__init__()
        
        # Hout=(Hin−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+2
        # default dilation = 1, output_padding = 0. with padding = 1, Hout = 2*Hin0
        self.conv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=use_bias)
        # for the output layer, sometimes we use 1 stride 1x1 kernel 
        #    self.conv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=1,
        #            stride=1, padding=0, bias=use_bias)
 
        # to set norm layer as identity use norm_layer=lambda x: Identity()
        self.norm_layer = norm_layer(output_nc)
        
        if non_linear == 'LeakyReLU':
            self.non_linear = nn.LeakyReLU(0.2, inplace=True)
        elif non_linear == 'Tanh':
            self.non_linear = nn.Tanh()
        elif non_linear == 'Sigmoid':
            self.non_linear = nn.Sigmoid()
        elif non_linear == 'ReLU':
            self.non_linear = nn.ReLU()
        elif non_linear == 'Identity':
            self.non_linear = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm_layer(x)
        x = self.non_linear(x) 
        return x

### gaussian smooth layer without learnable parameters
# https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/9
import math
import numbers

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = nnf.conv1d
        elif dim == 2:
            self.conv = nnf.conv2d
        elif dim == 3:
            self.conv = nnf.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

## usage
# smoothing = GaussianSmoothing(3, 5, 1)
# input = torch.rand(1, 3, 100, 100)
# input = nnf.pad(input, (2, 2, 2, 2), mode='reflect') # use reflection padding for the last two dimensions Width and Height
# output = smoothing(input)

class TPSLocator(nn.Module):
    """ 
        Locate the TPS control points locations
        input: relatedd higher order parameters
        output: control points
        Here we make this operate on a fixed prior, we can move the shape prior into input as well if needed
    """
    def __init__(self, input_dim, output_dim):
        super(TPSLocator, self).__init__()
        width = 32
        self.fc1 = nn.Linear(input_dim, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, output_dim)

    def robust_init(self, target_control_points):
        # Initialize the weights/bias with identity transformation
        # target_control_points are locations in the target image, a uniform grid
        # TODO check when will Pytorch init the network, see the link https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
        #bias = torch.from_numpy(np.arctanh(target_control_points.numpy())) # we use tanh to constrain the output to [-1,1]
        bias = torch.from_numpy(0.5*np.arctanh(target_control_points.numpy())) # we use tanh to constrain the output to [-1,1], but scale them by 2, to allow translation. So the input should be scale by 0.5 to make the init grid the same as a regular grid
        bias = bias.view(-1)
        self.fc3.bias.data.copy_(bias)
        self.fc3.weight.data.zero_()
        
    def forward(self, x):
        x = self.fc1(x)
        x = nnf.relu(x)
        
        x = self.fc2(x)
        x = nnf.relu(x)
        
        x = self.fc3(x)
        #x = nnf.tanh(x) # constrain the output to [-1, 1]
        x = 2*nnf.tanh(x) # constrain the output to [-2, 2], allow translation in larger area

        x = x.reshape(x.shape[0], -1, 2) # output x, y locations
        return x


def define_TPSLocator(input_dim, output_dim, target_control_points, init_type='normal', init_gain=0.02, gpu_ids=[], pretrained=None):
    """
    Create an simple linear layer
    
    Parameters:
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    """
    net = TPSLocator(input_dim, output_dim)
    net = init_net(net, init_type, init_gain, gpu_ids)
    # note we need to change the last layer here to have a robust start point
    net.module.robust_init(target_control_points) # init_net return a dataparallel object, which wraps the original nn module inside
    return net

class LinearConditionSampler(nn.Module):
    """ 
    linear latent output layer for CVAE condition sampling of z based on label c 
        1. change the latent dims to be aligned with covolutional features
        2. reshape the output dimensions
    """
    def __init__(self, input_dim, output_dim):
        super(LinearConditionSampler, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        x = self.fc(x)
        return x


def define_LinearConditionSampler(input_dim, output_dim, init_type='normal', init_gain=0.02, gpu_ids=[], pretrained=None):
    """
    Create an simple linear layer
    
    Parameters:
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    """
    net = LinearConditionSampler(input_dim, output_dim)
    return init_net(net, init_type, init_gain, gpu_ids)

## for solent18 1x1 cov net for Mutual information map
## reused in solent20 to generate the mask map
class PixelMap(nn.Module):
    """
    Create an 1x1 convolutional mapper, each pixel to the same label 

    a very simple CNN layer network, without linear layers
    fully control on the channel size, and last layer output type
    """

    def __init__(self, nc_factors, norm_layer=None, last_nonlinearFun='ReLU', printname='PixelMap'):
        """
        Parameters:
           norm_layer      -- normalization layer

        Output feature maybe a list of tensors in each layer instead of only the last layer tensor.
        """
        super(PixelMap, self).__init__()
        
        self.layers = []
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        for l in range(0, len(nc_factors)-2): # all but the last layer
            l_input_nc  = nc_factors[l]
            l_output_nc = nc_factors[l+1]    
            self.layers.append(PixelMapBlock(l_input_nc, l_output_nc, use_bias, norm_layer)) 
            self.add_module(printname+'_layer_'+str(l), self.layers[l])

        # final layer.         
        self.layers.append(PixelMapBlock(nc_factors[-2], nc_factors[-1], use_bias=True, norm_layer=lambda x: Identity(), non_linear=last_nonlinearFun)) 
        self.add_module(printname+'_layer_'+str(len(nc_factors)-1), self.layers[-1])

    def forward(self, x):
        for l in range(len(self.layers)):    
            x = self.layers[l](x)
        return x

def define_PixelMap(nc_factors, netDec=None, norm='none', last_nonlinearFun='ReLU', printname='PixelMap', init_type='normal', init_gain=0.02, gpu_ids=[], pretrained=None):

    """Create an 1x1 conv decoder

    NOTE! the default settings are used by the models. So be careful if you want to change them.

    unfinished:
    shall we merge enc-dec define?
    
    Parameters:
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        pretrained -- location of pretrained weights
    Returns an encoder

    The encoder has been initialized by <init_net>. It uses leaky RELU and tanh for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netDec == None:
        net = PixelMap(nc_factors, norm_layer=norm_layer, last_nonlinearFun=last_nonlinearFun, printname=printname)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netDec)
    return init_net(net, init_type, init_gain, gpu_ids)

class PixelMapBlock(nn.Module):
    """
    define the  block
    
    we changed the original design, to allow better control on the non-linear function

    """

    def __init__(self, input_nc, output_nc, use_bias, norm_layer, non_linear='LeakyReLU'):
        super(PixelMapBlock, self).__init__()
        
        # Hout=(Hin−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+2
        # default dilation = 1, output_padding = 0. with padding = 1, Hout = 2*Hin0
        self.conv = nn.Conv2d(input_nc, output_nc, kernel_size=1,
                stride=1, padding=0, bias=use_bias)
 
        # to set norm layer as identity use norm_layer=lambda x: Identity()
        self.norm_layer = norm_layer(output_nc)
        
        if non_linear == 'LeakyReLU':
            self.non_linear = nn.LeakyReLU(0.2, inplace=True)
        elif non_linear == 'Tanh':
            self.non_linear = nn.Tanh()
        elif non_linear == 'Sigmoid':
            self.non_linear = nn.Sigmoid()
        elif non_linear == 'ReLU':
            self.non_linear = nn.ReLU()
        elif non_linear == 'Identity':
            self.non_linear = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm_layer(x)
        x = self.non_linear(x) 
        return x

class MLFun(nn.Module):
    """ 
    a ML NN for any non-linear reshaping function  
    """
    def __init__(self, factors):
        super(MLFun, self).__init__()
        self.layers = []
        for l in range(len(factors)-1):
            nl = nn.Linear(factors[l], factors[l+1])
            self.layers.append(nl)
            self.add_module('layer'+str(l), nl)
    def forward(self, x):
        for l, nl in enumerate(self.layers):
            x = nl(x)
            # no relu for the last layer
            if l < len(self.layers)-1:
                x = nnf.relu(x) 
        return x


def define_MLFun(factors, init_type='normal', init_gain=0.02, gpu_ids=[], pretrained=None):
    """
    Create an simple linear layer
    
    Parameters:
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    """
    net = MLFun(factors)
    return init_net(net, init_type, init_gain, gpu_ids)

# start doge ==========================

# start MLP encoder  ---

class MLPEncoder_block(nn.Module):
    def __init__(self, lin, lout, non_linear_type):
        super().__init__()

        if non_linear_type=='relu':
            self.non_linear = nn.LeakyReLU(0.2) # nnf.relu()
        #self.non_linear = nn.Sigmoid()
        elif non_linear_type=='tanh':
            self.non_linear = nn.Tanh() #nnf.tanh()
        elif non_linear_type == 'Identity':
            self.non_linear = nn.Identity()
        else:
            raise NotImplementedError()

        self.fc=nn.Linear(lin, lout)

    def forward(self, x):
        """
        inputs: a batch of 1D tensor
        """
        x=self.fc(x)
        x=self.non_linear(x)

        return x

class MLPEncoder(nn.Module):
    """
    Defines a MLP encoder with width parameter
    
    do not use flattern x, assuming the x is aleady flattened
    """

    def __init__(self, l_width, non_linear_type):
        """
        l_width: [input_w, l1 , l2, ..., output_w]
        """
        super(MLPEncoder, self).__init__()
        
        self.layers = []
        # use sequential to save the non-linear setting to the model, so that we can use jit-scripts
        for i in range(len(l_width)-1):
            if i < len(l_width)-2:
                self.layers.append(MLPEncoder_block(l_width[i], l_width[i+1], non_linear_type))
            else: # for the last layer
                self.layers.append(MLPEncoder_block(l_width[i], l_width[i+1], "Identity")) 
            self.add_module('fc_layer'+str(i), self.layers[-1]) # add module is different from the layers[] list we keep TODO: refactor

    def forward(self, x):
        for l in range(len(self.layers)):    
            x = self.layers[l](x)
        return x

def define_MLPEncoder(l_width, init_type='normal', init_gain=0.02, gpu_ids=[], non_linear_type='relu'):
    """Create a mlp encoder

    Parameters:
        l_width(int list)  -- the width of each layer. Length is the layer number.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Our current implementation provides three types of discriminators:
    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = MLPEncoder(l_width, non_linear_type) 
    # raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


# --------------- individual LSO readout layers

class LSOReadout_v1(nn.Module):
    def __init__(self, in_units):
        """
            This is not used.
            
            init the weights to almost identical to y=x
        """
        super().__init__()
        # the initialisation is not ideal, but useful to illustrate our argument
        self.weight = nn.Parameter(torch.ones(in_units)*10) # range [0, 8]
        # self.weight1 = nn.Parameter(torch.zeros(in_units, units))
        # self.weight2 = nn.Parameter(torch.zeros(in_units, units))
        #self.bias = nn.Parameter(torch.ones(in_units,)*(-0.5)) # shift the range
        self.bias = nn.Parameter(torch.ones(in_units,)*(-5)) # shift the range

    def forward(self, lso):
        """
        lso are normalised
        """
        x = lso * self.weight + self.bias
        #return x @ self.weight + self.bias
        # x = torch.sigmoid(x) # leave this to the BCELoss_with_logits
        return x

class LSOReadout(nn.Module):
    def __init__(self, in_units):
        """
            init the weights to almost identical to y=x
        """
        super().__init__()
        # the initialisation is not ideal, but useful to illustrate our argument
        self.weight = nn.Parameter(torch.ones(1)*10) # range [0, 8]
        # self.weight1 = nn.Parameter(torch.zeros(in_units, units))
        # self.weight2 = nn.Parameter(torch.zeros(in_units, units))
        #self.bias = nn.Parameter(torch.ones(in_units,)*(-0.5)) # shift the range
        self.bias = nn.Parameter(torch.ones(1,)*(-5)) # shift the range

    def forward(self, lso):
        """
        lso are normalised
        """
        lso_m = lso.mean()
        x = lso_m * self.weight + self.bias
        #return x @ self.weight + self.bias
        # x = torch.sigmoid(x) # leave this to the BCELoss_with_logits
        return x

def define_LSOReadout(in_units, init_type, init_gain, gpu_ids=[], pretrained=None):
    """Create a mlp encoder

    remove the
        init_type (str)    -- the name of the initialization method.
    because we do not need that weight init

    Parameters:
        l_width(int list)  -- the width of each layer. Length is the layer number.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Our current implementation provides three types of discriminators:
    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = LSOReadout(int(in_units[0])) # convert string to int
    # raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, None, 0, gpu_ids) # no need to initialise the weights

class LSOMidline(nn.Module):
    
    def __init__(self, in_units):
        super().__init__()
        # the initialisation is not ideal, but useful to illustrate our argument
        self.weight_left = nn.Parameter(torch.ones(1)*10) # range [0, 1]
        self.bias_left = nn.Parameter(torch.ones(1,)*(-5)) # shift the range
        self.weight_right = nn.Parameter(torch.ones(1)*10) # range [0, 1]
        self.bias_right = nn.Parameter(torch.ones(1,)*(-5)) # shift the range

        # self.weight_m = nn.Parameter(torch.ones(1)*10) # range [0, 1]
        # self.bias_m = nn.Parameter(torch.ones(1,)*(-5)) # shift the range

    def forward(self, lso_left, lso_right):
        """
        lso are normalised

        lso: torch.Size([32]) 
        """
        # take mean of LSO reactions, [0, 1]
        # go through sigmoid to make sure the output is in [0, 1]
        # take the product of the two sigmoid outputs, with learnable parameters
        # directly output it is fine

        l1=lso_left.mean()
        l1=l1*self.weight_left+self.bias_left
        l1=torch.sigmoid(l1)
        
        r1=lso_right.mean() #*0.5, uni-lateral hearing loss should be on ILD, not here
        r1=r1*self.weight_right+self.bias_right
        r1=torch.sigmoid(r1)

        m=r1*l1
        
        # # layer one
        # l1 = lso_left * self.weight_left + self.bias_left
        # l1 = torch.sigmoid(l1) 
        # r1 = lso_right * self.weight_right + self.bias_right
        # r1 = torch.sigmoid(r1)
        # m = r1.mean()*l1.mean*()
        # m = self.weight_m * m + self.bias_m
        # # x = torch.sigmoid(x) # leave this to the BCELoss_with_logits
        return m


class LSOMidline_v0(nn.Module):
    
    def __init__(self, in_units):
        super().__init__()
        # the initialisation is not ideal, but useful to illustrate our argument
        self.weight_left = nn.Parameter(torch.ones(in_units)*10) # range [0, 1]
        self.bias_left = nn.Parameter(torch.ones(in_units,)*(-5)) # shift the range
        self.weight_right = nn.Parameter(torch.ones(in_units)*10) # range [0, 1]
        self.bias_right = nn.Parameter(torch.ones(in_units,)*(-5)) # shift the range

        self.weight_m = nn.Parameter(torch.ones(in_units)*10) # range [0, 1]
        self.bias_m = nn.Parameter(torch.ones(in_units,)*(-5)) # shift the range

    def forward(self, lso_left, lso_right):
        """
        lso are normalised
        """
        # layer one
        l1 = lso_left * self.weight_left + self.bias_left
        l1 = torch.sigmoid(l1) 
        r1 = lso_right * self.weight_right + self.bias_right
        r1 = torch.sigmoid(r1)
        m = r1*l1
        m = self.weight_m * m + self.bias_m
        # x = torch.sigmoid(x) # leave this to the BCELoss_with_logits
        return m
    

def define_LSOMidline(in_units, init_type, init_gain, gpu_ids=[], pretrained=None):
    """Create a mlp encoder

    remove the
        init_type (str)    -- the name of the initialization method.
    because we do not need that weight init

    Parameters:
        l_width(int list)  -- the width of each layer. Length is the layer number.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Our current implementation provides three types of discriminators:
    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = LSOMidline(int(in_units[0])) # convert string to int
    # raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, None, 0, gpu_ids) # no need to initialise the weights

# --------------- permutation invariant layers
class PermEqui2_mean(nn.Module):
  """
  copied code from https://github.com/manzilzaheer/DeepSets/blob/master/PointClouds/classifier.py

  use the fact that pytorch broadcast the input 
    Input: (∗,Hin)(*, H_{in})(∗,Hin​) where ∗*∗ means any number of dimensions including none and Hin=in_featuresH_{in} = \text{in\_features}Hin​=in_features.
  """
  def __init__(self, in_dim, out_dim):
    super(PermEqui2_mean, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)
    self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

  def forward(self, x):
    xm = x.mean(1, keepdim=True)
    xm = self.Lambda(xm) 
    x = self.Gamma(x)
    x = x - xm
    return x

class permuinvaMLPEncoder(nn.Module):
    """
    Defines a perumation invariant MLP encoder with width parameter
    
    do not use flattern x, assuming the x is aleady flattened
    """

    def __init__(self, l_width, non_linear_type):
        """
        l_width: [input_w, l1 , l2, ..., output_w]
        """
        super(permuinvaMLPEncoder, self).__init__()
        
        self.layers = []
        for i in range(len(l_width)-1):
            self.layers.append(PermEqui2_mean(l_width[i], l_width[i+1])) 
            self.add_module('permutationEquiMean_layer'+str(i), self.layers[-1])

        if non_linear_type=='relu':
            self.non_linear = nn.LeakyReLU(0.2)
        #self.non_linear = nn.Sigmoid()
        elif non_linear_type=='tanh':
            self.non_linear = nn.Tanh()
        else:
            raise NotImplementedError()

    def forward(self, x):
        """
        x: a batch of 2D tensor, which is a 3D tensor by itself [batch_size, set_width, feature_width]
        
        The trick is to combine the batch dimension with the  
        """
        # allow batchsize > 1, no need for flatterning here, torch can handle the extra empty deminsions.  #x = flattern_tensor(inputs)
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            x = self.non_linear(x)

        x = self.layers[-1](x)
        #x = nnf.sigmoid(x) # do not use this sigmoid. we use CrossEntropyLoss in classifier 

        return x

def define_permuinvaMLPEncoder(l_width, init_type='normal', init_gain=0.02, gpu_ids=[], non_linear_type='relu'):
    """Create a permutation invariant mlp encoder

    Parameters:
        l_width(int list)  -- the width of each layer. Length is the layer number.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Our current implementation provides three types of discriminators:
    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = permuinvaMLPEncoder(l_width, non_linear_type) 
    # raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)

# start dropout MLP encoder  ---

class dropoutMLPEncoder(nn.Module):
    """
    Defines a MLP encoder with width parameter
    
    do not use flattern x, assuming the x is aleady flattened
    """

    def __init__(self, l_width, non_linear_type, dropout_p):
        """
        l_width: [input_w, l1 , l2, ..., output_w]
        """
        super(dropoutMLPEncoder, self).__init__()
        
        self.layers = []
        for i in range(len(l_width)-1):
            layer=nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(l_width[i], l_width[i+1])) 
            self.layers.append(layer) 
            self.add_module('dropout+fc_layer'+str(i), self.layers[-1])

        if non_linear_type=='relu':
            self.non_linear = nn.LeakyReLU(0.2)
        #self.non_linear = nn.Sigmoid()
        elif non_linear_type=='tanh':
            self.non_linear = nn.Tanh()
        else:
            raise NotImplementedError()
        

    def forward(self, inputs):
        """
        inputs: a batch of 1D tensor
        """
        x = inputs 
        # allow batchsize > 1, no need for flatterning here, torch can handle the extra empty deminsions.  #x = flattern_tensor(inputs)
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            x = self.non_linear(x)

        x = self.layers[-1](x)
        #x = nnf.sigmoid(x) # do not use this sigmoid. we use CrossEntropyLoss in classifier 

        return x

def define_dropoutMLPEncoder(l_width, init_type='normal', init_gain=0.02, gpu_ids=[], non_linear_type='relu', dropout_p=0.0):
    """Create a mlp encoder

    Parameters:
        l_width(int list)  -- the width of each layer. Length is the layer number.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Our current implementation provides three types of discriminators:
    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = dropoutMLPEncoder(l_width, non_linear_type, dropout_p) 
    # raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)

# ----------- doge49


###########################
# TODO better definition here, so that jit can work.
#############################
# #@torch.jit.script  # this gives errors?
# class FullConvNet(nn.Module):
#     """
#     a very simple CNN layer network, without linear layers
#     fully control on the channel size, and last layer output type
#     """

#     def __init__(self, config_list:list):
#         """Construct an conv encoder with attention input

#         to configure a fullConvNet

#         Note!
#         We change nc_factors here from reverse the encoder to independent and normal order.
#         Including the input channel number and output channel number, it is a FCN

#         TODO: change the definition with list of parameters

#         Parameters:
#            norm_layer      -- normalization layer

#         Output feature maybe a list of tensors in each layer instead of only the last layer tensor.
#         """
#         super(FullConvNet, self).__init__()
        
#         self.layers = []
#         #self.layers: List[nn.Module] = [] # still can not be used with torch.jit
#         #self.layers=nn.ModuleList() # still does not work, with a new error: index need to be int
        
        
#         # we name the layers in decoder reversely, so that pretrain enc-dec from outside-in and preload
#         # if padding=1, w_output = w_input*2; if padding=0, w_output = w_input*2+2. totally reverse of the convolution padding. we need padding for the inner most layer, othewise, although the kernel size is 4, output is 2x2
#         for l,c in enumerate(config_list): # all but the last layer
#             self.layers.append(FullConvNetBlock(c)) 
#             #self.add_module('layer_'+str(l), self.layers[l])
#         self.layers=nn.Sequential(*self.layers)
#         #return self.layers # gives error __init__() should return None, not 'Sequential'


#         # final layer.         
#         #self.layers.append(FullConvNetBlock(nc_factors[-2], nc_factors[-1], use_bias=True, norm_layer=lambda x: Identity(), stride=last_stride, padding=last_padding, non_linear=last_nonlinearFun, kernel_size=last_kernel)) 
#         #self.add_module(printname+'_layer_'+str(len(nc_factors)-1), self.layers[-1])

#     # def forward(self, x):
#     #     for l in range(len(self.layers)):    
#     #         x = self.layers[l](x)
#     #     return x

#@torch.jit.script  # this gives errors?
class FullConvNet(nn.Module):
    """
    a very simple CNN layer network, without linear layers
    fully control on the channel size, and last layer output type
    """

    def __init__(self, config_list:list):
        """Construct an conv encoder with attention input

        to configure a fullConvNet

        Note!
        We change nc_factors here from reverse the encoder to independent and normal order.
        Including the input channel number and output channel number, it is a FCN

        TODO: change the definition with list of parameters

        Parameters:
           norm_layer      -- normalization layer

        Output feature maybe a list of tensors in each layer instead of only the last layer tensor.
        """
        super(FullConvNet, self).__init__()
        
        self.layers = []
        #self.layers: List[nn.Module] = [] # still can not be used with torch.jit
        #self.layers=nn.ModuleList() # still does not work, with a new error: index need to be int
        
        # we name the layers in decoder reversely, so that pretrain enc-dec from outside-in and preload
        # if padding=1, w_output = w_input*2; if padding=0, w_output = w_input*2+2. totally reverse of the convolution padding. we need padding for the inner most layer, othewise, although the kernel size is 4, output is 2x2
        for l,c in enumerate(config_list): # all but the last layer
            self.layers.append(FullConvNetBlock(c)) 
            self.add_module('layer_'+str(l), self.layers[l])
        #self.layers=nn.Sequential(*self.layers)
        #return self.layers # gives error __init__() should return None, not 'Sequential'


        # final layer.         
        #self.layers.append(FullConvNetBlock(nc_factors[-2], nc_factors[-1], use_bias=True, norm_layer=lambda x: Identity(), stride=last_stride, padding=last_padding, non_linear=last_nonlinearFun, kernel_size=last_kernel)) 
        #self.add_module(printname+'_layer_'+str(len(nc_factors)-1), self.layers[-1])

    def forward(self, x):
        for l in range(len(self.layers)):    
            x = self.layers[l](x)
        return x

class FullConvNetBlock(nn.Module):
    """
    define the decoder block


    in-channel, out-channel, conv type(transpose 0/1), kernel size, padding, stride, non-linear, (bias)
    i3-o4-t0-k3-p0-s2-fRelu 
    q: output padding for trans-conv2d
    
    input_nc, output_nc, use_bias, non_linear='LeakyReLU', stride=2, padding=1, kernel_size=4
    """

    def __init__(self, config):
        super(FullConvNetBlock, self).__init__()
        
        # TODO add syntax check here
        # what if we want to add in new arguments?
        # what if there is a missing argument?
        # where can we find the dictionary? 
        # what if the names are conflicting?
        # and the naming is bad: only allow a single letter argument, e.g. outputpadding op is conflicting with o
        configs={}
        for k in config.split('-'):
            if k[0]!='f': # take the int value
                configs[k[0]]=int(k[1:])
            else: # take the string value
                configs[k[0]]=k[1:]

        if configs['t']==1:
            if not 'q' in configs.keys():
                configs['q']=0
            # transposed conv
            # check with the pytorch mamual
            # Hout​=(Hin​−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
            # default dilation = 1, output_padding = 0. 
            self.conv = nn.ConvTranspose2d(configs['i'], configs['o'], kernel_size=configs['k'],
                    stride=configs['s'], padding=configs['p'], output_padding=configs['q'], bias=True)
            # for the output layer, sometimes we use 1 stride 1x1 kernel 
            #    self.conv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=1,
            #            stride=1, padding=0, bias=use_bias)
        else:
            # pytorch manual
            # Hout​=⌊(Hin​+2×padding[0]−dilation[0]×(kernel_size[0]−1)−1​)/stride[0]+1⌋
            self.conv = nn.Conv2d(configs['i'], configs['o'], kernel_size=configs['k'],
                    stride=configs['s'], padding=configs['p'], bias=True)

        non_linear=configs['f']
        if non_linear == 'LeakyReLU':
            self.non_linear = nn.LeakyReLU(0.2, inplace=True)
        elif non_linear == 'Tanh':
            self.non_linear = nn.Tanh()
        elif non_linear == 'Sigmoid':
            self.non_linear = nn.Sigmoid()
        elif non_linear == 'ReLU':
            self.non_linear = nn.ReLU()
        elif non_linear == 'Identity':
            self.non_linear = nn.Identity()
        else:
            raise NotImplementedError('unknown non-linear configuration')
        
    def forward(self, x):
        x = self.conv(x)
        x = self.non_linear(x) 
        return x

def define_FullConvNet(config_list, init_type='normal', init_gain=0.02, gpu_ids=[], **kwargs):

    """Create an decoder

    NOTE! the default settings are used by the models. So be careful if you want to change them.

    unfinished:
    1. shall we merge enc-dec define?
    2. TODO: should simplify the code here

    Parameters:
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        pretrained -- location of pretrained weights
    Returns an encoder

    The encoder has been initialized by <init_net>. It uses leaky RELU and tanh for non-linearity.
    """
    net = None

    net = FullConvNet(config_list)
    return init_net(net, init_type, init_gain, gpu_ids)

# TODO:
# 1. merge the options for defining conv\deconv\mlp

# ----------------- 
# picasso project:
#           Conv U-Net

class ConvUNet(nn.Module):
    """
    full conv

    key is to have a diff forward()

    # TODO:
        1. here we rely on the user to make sure that the network architecture works well: 
            combined x and saved map have the same dimension
            chanel size are OK
    """

    def __init__(self, config_list:list):
        """Construct an conv encoder with attention input

        to configure a fullConvNet

        Note!
        We change nc_factors here from reverse the encoder to independent and normal order.
        Including the input channel number and output channel number, it is a FCN

        TODO: change the definition with list of parameters

        Parameters:
           norm_layer      -- normalization layer

        Output feature maybe a list of tensors in each layer instead of only the last layer tensor.
        """
        super(ConvUNet, self).__init__()
        
        self.layers = []
        #self.layers: List[nn.Module] = [] # still can not be used with torch.jit
        #self.layers=nn.ModuleList() # still does not work, with a new error: index need to be int
        
        # we name the layers in decoder reversely, so that pretrain enc-dec from outside-in and preload
        # if padding=1, w_output = w_input*2; if padding=0, w_output = w_input*2+2. totally reverse of the convolution padding. we need padding for the inner most layer, othewise, although the kernel size is 4, output is 2x2
        for l,c in enumerate(config_list): # all but the last layer
            self.layers.append(FullConvNetBlock(c)) 
            self.add_module('layer_'+str(l), self.layers[l])
        #self.layers=nn.Sequential(*self.layers)
        #return self.layers # gives error __init__() should return None, not 'Sequential'


        # final layer.         
        #self.layers.append(FullConvNetBlock(nc_factors[-2], nc_factors[-1], use_bias=True, norm_layer=lambda x: Identity(), stride=last_stride, padding=last_padding, non_linear=last_nonlinearFun, kernel_size=last_kernel)) 
        #self.add_module(printname+'_layer_'+str(len(nc_factors)-1), self.layers[-1])

        self.depth=int((l+1)/2)

    def forward(self, x):
        # downward pass
        down_maps=[] 
        for l in range(self.depth-1):    
            x = self.layers[l](x)
            down_maps.append(x)
        
        # bottom pass, no save
        x = self.layers[self.depth-1](x)
        x = self.layers[self.depth](x)

        # upward pass
        for l in range(self.depth+1, self.depth*2):
            # depth
            d = l-self.depth
            saved_x = down_maps[-d]
            combined_x=torch.cat([saved_x, x], dim=1) # x [batch, channel, w, h]
            x = self.layers[l](combined_x)

        return x


def define_ConvUNet(config_list, init_type='normal', init_gain=0.02, gpu_ids=[], **kwargs):
    """Create a conv U-net

    NOTE! the default settings are used by the models. So be careful if you want to change them.

    Parameters:
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        pretrained -- location of pretrained weights
    Returns an encoder

    The encoder has been initialized by <init_net>. It uses leaky RELU and tanh for non-linearity.
    """
    net = None
    net = ConvUNet(config_list)
    return init_net(net, init_type, init_gain, gpu_ids)
