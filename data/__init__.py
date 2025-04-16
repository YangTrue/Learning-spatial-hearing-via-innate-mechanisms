"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from ginvae.data.base_dataset import BaseDataset


import torch
# https://github.com/pytorch/pytorch/issues/11201
# to avoid the too many opened files error when batch number is large
torch.multiprocessing.set_sharing_strategy('file_system')

def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "ginvae.data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """
    Wrapper class of Dataset class that performs multi-threaded data loading
    TODO: improve the pattern here, allow more flexible operations on the dataset flow, but also work with paralell process.

        - allow explicit choice of dataset loading method, collate_fn
    """

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        self._update_loader()
        print("dataset [%s] was created" % type(self.dataset).__name__)
    
    def _update_loader(self):
        if 'bezier' in self.opt.dataset_mode:
            # bezier experiments need various length items in a batch
            # reuse the built-in dataloader, because it is good for sampling and pin memory and so on
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                collate_fn=lambda x:x,
                batch_size=self.opt.batch_size,
                shuffle=not self.opt.serial_batches,
                num_workers=0) 
                # set the num_workers to 0 means loading from the main process 
                #num_workers=int(self.opt.num_threads))
        else:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.opt.batch_size,
                shuffle=not self.opt.serial_batches,
                num_workers=int(self.opt.num_threads))

    def restart(self):
        self.dataset.restart()
        self._update_loader()
            
    def val_restart(self):
        self.dataset.val_restart()
        self._update_loader()
 
    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            #if i * self.opt.batch_size >= self.opt.max_dataset_size:
            # when used for Validation, the max_dataset_size in option is about training dataset size
            if i * self.opt.batch_size >= min(len(self.dataset), self.opt.max_dataset_size):
                break
            yield data
