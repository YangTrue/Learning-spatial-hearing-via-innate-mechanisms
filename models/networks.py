import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as nnf

###############################################################################
# Helper Functions
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

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
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


class AttentionEncoder(nn.Module):
    """Create an convolutional encoder with potential attention input"""

    def __init__(self, input_nc, num_downs=4, ngf=4, norm_layer=None, use_dropout=False):
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
    
        #nc_factors = [1, 2, 8, 32, 16]
        nc_factors = [1, 2, 4, 32]
        #nc_factors = [1, 2, 4, 64] # increase the size so that get more details when replace tanh() with sigmoid()

        assert ngf==4, "first layer channel should be fixed as 4"
        assert num_downs<=4, "no more than 4 layers. we replace relu with tanh/sigmoid in 4th layer"

        self.layers = []

        #if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
        #    use_bias = norm_layer.func == nn.InstanceNorm2d
        #else:
        #    use_bias = norm_layer == nn.InstanceNorm2d
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        
        self.layers.append(AttentionEncoderBlock(input_nc, ngf, use_bias, norm_layer, 0))
        self.add_module('encoder_layer_'+str(0), self.layers[0]) # registry, otherwise not included module
        for l in range(1, num_downs):
            l_input_nc  = ngf * nc_factors[min(len(nc_factors)-1, l-1)]
            l_output_nc = ngf * nc_factors[min(len(nc_factors)-1, l  )]    
            self.layers.append(AttentionEncoderBlock(l_input_nc, l_output_nc, use_bias, norm_layer, l))
            self.add_module('encoder_layer_'+str(l), self.layers[l])
        #self.layers[-1].l = -1 #set the innermost layer # deprecated, we count the layers explicitly

    def forward(self, x, attention_flag=False, channelwise_attention=None):
        output = []
        for l in range(len(self.layers)):   
            if attention_flag and not channelwise_attention==None:
                l_attention = channelwise_attention[l]
            else:
                l_attention = None
            x, avg_act = self.layers[l](x, attention_flag, l_attention, attention_scaler=None)
            output.append((x, avg_act)) 
        return output

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
        else:
            self.non_linear = nn.Tanh()
            #self.non_linear = nn.Sigmoid()
            padding = 0

        # we fix padding to 0, note this must compatible with output
        self.conv = nn.Conv2d(input_nc, output_nc, kernel_size=4,
                stride=2, padding=padding, bias=use_bias)
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
        
        if self.l != 0:
            #x = self.relu(x)
            x = self.non_linear(x)
        
        x = self.conv(x)
        if attention_flag:
            x = self.expand_attention(x, channelwise_attention, attention_scaler)
        
        if self.l != 0 and self.l != 3: # not the last layer in the 4-layer encoder 
            x = self.norm_layer(x) # unfinished: this hurt the 1x1 conv
            
        #get average reaction, [batch, channel]
        #may also use max or min or any other ones.
        avg_act = x.mean(-1).mean(-1)

        return x, avg_act

class Decoder(nn.Module):
    """Create an convolutional decoder"""

    def __init__(self, output_nc, num_downs=4, ngf=64, norm_layer=None, use_dropout=False):
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
    
        #nc_factors = [1, 2, 8, 32, 16]
        nc_factors = [1, 2, 4, 32]
        #nc_factors = [1, 2, 4, 64]
        self.layers = []

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
            self.layers.append(DecoderBlock(l_input_nc, l_output_nc, use_bias, norm_layer, l_inverse, padding=l_padding))# add padding for 32 example
            self.add_module('decoder_layer_'+str(l_inverse), self.layers[l])
        self.layers.append(DecoderBlock(ngf, output_nc, use_bias=True, norm_layer=lambda x: Identity(), l=0, padding=1)) # output layer, extra padding for the 48x48 images
        self.add_module('decoder_layer_'+str(0), self.layers[num_downs-1])

    def forward(self, x):
        for l in range(len(self.layers)):    
            x = self.layers[l](x)
        return x

class DecoderBlock(nn.Module):
    """define the decoder block"""

    def __init__(self, input_nc, output_nc, use_bias, norm_layer, l, padding=0):
        super(DecoderBlock, self).__init__()
        
        self.conv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4,
                stride=2, padding=padding, bias=use_bias)
        self.norm_layer = norm_layer(output_nc)
        self.l = l

    def forward(self, x):
        x = nnf.relu(x)
        x = self.conv(x)
         
        if self.l == 0:
        # outer most layer
            x = nnf.tanh(x)
        else:
            x = self.norm_layer(x)
        
        return x


def define_Encoder(input_nc, ngf, num_downs=4, netEnc=None, norm='none', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], pretrained=None):
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
        net = AttentionEncoder(input_nc, num_downs=num_downs, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netEnc)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_Decoder(output_nc, ngf, num_downs=4, netDec=None, norm='none', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], pretrained=None):
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
        net = Decoder(output_nc, num_downs=num_downs, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout)
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

        assert input_t_length == input_c_length, "top-down and bottom-up code length should be the same"
        assert input_c_length == output_c_length, "output code is only a changed version of input"
        
        input_length = input_t_length*3
        self.fc1 = nn.Linear(input_length, input_length*2)
        self.fc15 = nn.Linear(input_length*2, input_length*2)
        self.fc2 = nn.Linear(input_length*2, output_c_length)
        
    def forward(self, m, c, t):
        """
        t: top-down code
        c: bottom-up code
        m: middle ground target to optimiz on
        """
        if len(t.size()) > 2:
            flatten_t = t.reshape(1, -1)
        else:
            flatten_t = t
        
        if len(m.size()) > 2:
            flatten_m = m.reshape(1, -1)
        else:
            flatten_m = m 

        flatten_c = c.reshape(1, -1)

        x = torch.cat((flatten_c, flatten_t, flatten_m), 1)
        x = self.fc1(x)
        
        x = nnf.relu(x) 
        x = self.fc15(x)

        x = nnf.tanh(x) # use tanh() here to be comparable with the original output code
        #x = nnf.sigmoid(x) # use tanh() here to be comparable with the original output code
        
        x = self.fc2(x)
        # no relu here, because the input x is not relued
        #x = nnf.relu(x)
        x = x.reshape(1, 128, 1, 1)
        #x = x.reshape(1, 256, 1, 1) # cooperate with sigmoid
        
        return x


