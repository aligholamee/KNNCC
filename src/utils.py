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

def load_cifar(path):
    """
    Loads the CIFAR Image Dataset
    :param path: string
    :return: given dataset path in a numpy array
    """

    # Get the image filelist
    image_list = glob.glob(path)

    # Load the images one by one
    data = np.array([np.array(Image.open(image)) for image in image_list])

    return data

