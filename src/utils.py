# ========================================
# [] File Name : utils.py
#
# [] Creation Date : April 2018
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================

import glob
import numpy as np
from PIL import Image
import pickle

def load_cifar(path):
    """
    Loads the CIFAR Image Dataset
    :param path: string
    :return: given dataset path in a numpy array
    """
    with open(path, 'rb') as batch_1:
        loaded = pickle.load(batch_1, encoding='bytes')

    return loaded[b'data'],np.array(loaded[b'labels'])

def store_image(data, path, index):
    """
    Stores the images of type numpy to the path
    :param data: numpy array
    :param path: string
    :return: string
    """
    img = Image.fromarray(data, 'RGB')
    img.save(index + '.png')
    img.show()

    return "Saved" + index
