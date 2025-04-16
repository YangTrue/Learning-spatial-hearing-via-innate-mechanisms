"""This package contains modules related to objective functions, optimizations, and network architectures.

To add a custom model class called 'dummy', you need to add a file called 'dummy_model.py' and define a subclass DummyModel inherited from BaseModel.
You need to implement the following five functions:
    -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
    -- <set_input>:                     unpack data from dataset and apply preprocessing.
    -- <forward>:                       produce intermediate results.
    -- <optimize_parameters>:           calculate loss, gradients, and update network weights.
    -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.

In the function <__init__>, you need to define four lists:
    -- self.loss_names (str list):          specify the training losses that you want to plot and save.
    -- self.model_names (str list):         define networks used in our training.
    -- self.visual_names (str list):        specify the images that you want to display and save.
    -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an usage.

Now you can use the model class by specifying flag '--model dummy'.
See our template model class 'template_model.py' for more details.
"""

import importlib
from ginvae.models.base_model import BaseModel
from ginvae.models.base_runner import BaseRunner


def find_module_using_name(module_name, module_type):
    """Import the module "modules/[module_name]_[module_type].py".

    module_type: model or runner

    In the file, the class called DatasetNameModel() will
    be instantiated. 
    and it is case-insensitive.
    """
    module_filename = "ginvae.models." + module_name + "_"+module_type
    modulelib = importlib.import_module(module_filename)
    module = None
    target_module_name = module_name.replace('_', '') + module_type
    for name, cls in modulelib.__dict__.items():
        if name.lower() == target_module_name.lower() \
           and (issubclass(cls, BaseModel) or issubclass(cls, BaseRunner)):
            module = cls

    if module is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (module_filename, target_module_name))
        exit(0)

    return module


def get_option_setter(model_name, module_type='model'):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_module_using_name(model_name, module_type)
    return model_class.modify_commandline_options


def create_model(opt):
    """Obseleted code: not used anymore. replaced by the create_module() call
    
    Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    model = find_module_using_name(opt.model,'model')
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance

def create_module(opt, module_name, module_type, tag):
    """Create a module given the option.
    
    NOTE: the ba thing is that we lose the automatic reference link in VS code. Because the compiler can not find the link back to the definition before running
    Can we pass that information? Keep the name-indexed advantage at the sametime?

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    module = find_module_using_name(module_name, module_type)
    # TODO: manage the options here, make sure that all the tag and options are not overlapped
    instance = module(opt, tag)
    print("[%s] [%s] was created" %(module_type, type(instance).__name__))
    return instance