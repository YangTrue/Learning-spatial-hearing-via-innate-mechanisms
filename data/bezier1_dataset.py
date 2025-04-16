from ginvae.data.base_dataset import BaseDataset
#from ginvae.data.image_folder import make_dataset
from PIL import Image
import os
import pickle

import numpy as np
import math 
import torch

# add in the pickle extension

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.pickle'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    """
    since sorted takes too long, we remove it here and immediately stop the walking
    """
    print("making image dataset from folder "+dir)
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    
    image_count = 0
    for root, _, fnames in os.walk(dir):
    #for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path) # unfinished: can we make this faster?
                image_count += 1
                if image_count >= max_dataset_size:
                    return images
    #return images[:min(max_dataset_size, len(images))]
    return images

class Bezier1Dataset(BaseDataset):
    """

    230730 Bezier 1

    change from single large pickle file to dataset directory
    
    ------------- bezier0
    a simple mnist dataset for project PICASSO.
    remove domain adaptation DOGE project

    assumptions and decisions
        1. pickle file input, a list of image item and meta information
            - use grey iamge for now

    future tasks:
        1. add unique nametags for each dataset object

    # ------- Old comments from Chairs
    build based on abide dataset.
    Instead of 3 levels Sites(domain)-Persons-Frames
    we use 3 levels for Chairs(domain)-Views

    design decisions
    [] test/train phase
    [] test/train split
    [] cutoff batch size and end of dataset
    [] transformation of data value: we use the whole dataset min, max. A bit off from realistic settings, can change.
    [] balanced sampling for classes y? inside each domain d
    [Y] random sub-set domain sampling? It is possible, but we move the pointer globally, so there may be data that is not used.
    [x] move whole dataset to the GPU. too much to change

    lessons
    - multi-thread breaks pdb debugging, even if you use single child thread
    
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        """
        parser.add_argument('--image_w', type=int, default=105,  help='image width')


        # TODO remove this option. how to do this? coupled with the validator settings now. 
        parser.add_argument('--val_repeat_sampling', type=int, default=1, help='repeat sampling during evaluation')

        return parser


    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        
        TODO:
            refactor to explicit phase choice for the init function?
        """
        BaseDataset.__init__(self, opt)
        
        #assert opt.num_threads == 1 # ~~more than 1 threads are WRONG!~~ now we allow them
                
        # no need to move to the device, leave that job to set_inut
        #self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU

        self.phase_code=opt.phase

        # CHU
        # TODO: allow choice of input file, from director. separate that option from the method to init the dataset
        # TODO: allow diff file names
        # TODO: dynamic options? if we add a certain tag, we design the datasets? e.g. give the dataset file path, build the loader and call it?
        # here we use implicit rules/contracts between dataset naming and program
        if opt.phase in ['train', 'val', 'test']:
                #with open(os.path.join(opt.dataroot, opt.phase+'_data.pickle'), 'rb') as fp:
                #    df=pickle.load(fp)
                target_dir = os.path.join(opt.dataroot, opt.phase)
                df=make_dataset(target_dir, max_dataset_size=opt.max_dataset_size) # TODO implict cotract, how to hornor this ?
        else:
            raise ValueError('Unknown phase')

        self.data = df
        return
    

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        hidden contract with dataloader and director:
            1. input data should be tensor, no need to convert from numpy
            2. the input cpts tensor can have diff shapes in a batch, we simply use dataloader to sample from the datasets, but not merging them into large tensor item

        # ---------------- old

        # do no use torchvision.transform, do our own
        # to tensor, (normalise)

        # something unknown happened here inside the multi-thread pytorch.
        # they convert the [batch of tuples] to [tuples of batch], so the result we got from the other end
        # is [batch tuple of element 1, tuple of element 2]
 
        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        item_path = self.data[index]
        with open(item_path, 'rb') as fp:
            item=pickle.load(fp)
        #item = (0 x_img, 1 motor_splines, 2 None, 3 x_t_img, 4 dot_img, 5 tangent_img, 6 s_img, 7 next_sub) # 
        # each item is ( (hw, hw), bezier shape tensor(ncpt, 2) )
        # s is target stroke
        #  item = (x_img, motor_splines, None, x_t_img, None, None, None, None, 8 unmask_end_img, 9 unmask_end_points)
        # TODO refactor
        #   what if missing item? using more flexible ways to organise it. Using dictionary instead of list/tuples

        # example of converting item by item
        if item[7] is None:
            nsdp=0 # next sub degree plus. None means the end  point of a stroke
        else:
            nsdp=item[7].shape[0] # do not -1, here we PLUS 1 for all degree, leave d=0 as the end point symbol

        if len(item)>8:
            return {'x':item[0], 'x_t':item[3], 'dot':item[4], 'tangent':item[5],'sub_motor_cpts':item[7], 'unmask_end_img':item[8], 'unmask_end_points':item[9], 'nsdp':nsdp} 
        else:
            return {'x':item[0], 'x_t':item[3], 'dot':item[4], 'tangent':item[5],'sub_motor_cpts':item[7], 'nsdp':nsdp} 

    def __len__(self):
        return len(self.data)
    
    def restart(self):
        pass
