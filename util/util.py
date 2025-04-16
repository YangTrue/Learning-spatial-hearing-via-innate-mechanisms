"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


def color_overlay(foreground, background, color_f=[0, 255, 0], color_b=[100, 100, 100]):
    """
    target, background: RGB numpy arrays, assume each channel are the same--originally be grayscale, converted by the tensor2im() function 
    return:
    combined: same RGB numpy arrays
    """
    assert(foreground.size == background.size)
    combined = np.zeros_like(foreground)
    for i in range(foreground.shape[0]):
        for j in range(foreground.shape[1]):
            if foreground[i, j, 0] == 0: # black foreground:
                combined[i, j, :] = np.round(background[i, j, :]*color_b/255.0).astype(np.uint8)
            elif background[i, j, 0] == 0:
                combined[i, j, :] = np.round(foreground[i, j, :]*color_f/255.0).astype(np.uint8)
            else:
                #combined[i, j, :] = np.round(np.minimum(foreground[i, j, :]*color_f, background[i, j, :]*color_b)/255.0).astype(np.uint8)
                #combined[i, j, :] = np.round(np.maximum(foreground[i, j, :]*color_f, background[i, j, :]*color_b)/255.0).astype(np.uint8) # does not work, the forground are not visable
                #combined[i, j, :] = np.round(foreground[i, j, :]*color_f/255.0).astype(np.uint8)
                combined[i, j, :] = np.round((0.9*foreground[i, j, :]*color_f + 0.1*background[i, j, :]*color_b)/255.0).astype(np.uint8)
    return combined

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1)) # repeat in channel, do not resize in wxh
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

import torchvision
import PIL

def save_tensor_list_to_row_aglined_grid_image(tl, dst):
    """
    input: 
        tl -- list of tensors. Each tensor is a 1D list of images [NxCxHxW], of the same length. robust to: detached, at cpu
        dst -- destination path for saving the image
    output: 
        img -- a png file, row for each element in tl, col for each sample

    goal: 
        used for comparing the swap results
        row1: input a
        row2: input b
        row3: output mixture of a,b
    """
    tl=[tin.detach().cpu() for tin in tl]
    n=tl[0].shape[0]
    tl=torch.cat(tl, 0) 
    img=np.moveaxis(torchvision.utils.make_grid(tl, nrow=n).numpy(),[0],[2])
    img=(img*255).astype(np.uint8) # otherwise, PIL will fail
    img=PIL.Image.fromarray(img)
    img.save(dst)
    return


