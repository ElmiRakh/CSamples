import os
from skimage.io import imread
from skimage import img_as_ubyte,img_as_float,exposure,color
import numpy as np
from tqdm import tqdm
import sys


indx = 250
def load_images(path,idx,count):
    images = []
    for fname in os.listdir(path)[idx:idx+count]:

        fname = os.path.join(path, fname)
        image = imread(fname) / 255.
        images.append(np.atleast_3d(image))

    return images

def batch_generator(path_x,batch_size,max_steps,bin_multiplier=2):
    global indx
    while True:
        
        X = np.array(load_images(path_x,indx,batch_size))
        Y = np.array(load_images('./data/B80/train',indx,batch_size))
        if indx >= max_steps:
            indx = 0
        else:
            indx += 1

        yield (X,Y)

def load_single_pair(path_x,idx,count):
    X = load_images(path_x,idx,count)
    Y = load_images('./data/B80/train',idx,count) 
    return (X,Y)

