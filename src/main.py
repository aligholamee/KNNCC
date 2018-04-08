# ========================================
# [] File Name : main.py
#
# [] Creation Date : April 2018
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================

# from .knn import NearestNeighbour as NN
from utils import load_cifar
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Data root path
DATA_ROOT = './data/'
BATCH_LIST = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
STORE_IMAGES = []
# Load the data
train_data = np.array([])
train_labels = np.array([])

for batch in BATCH_LIST:
    print(batch)
    data, labels = load_cifar(DATA_ROOT + batch)
    print(data.shape)
    print(labels.shape)
    train_data = np.append(train_data, data)
    train_labels = np.append(train_labels, labels)

image_set = train_data.reshape(50000, 3072)
for image in image_set:
    red_channel = image[0:1024]
    red_channel = red_channel.reshape(32, 32)
    green_channel = image[1025: 2048]
    green_channel = green_channel.reshape(32, 32)
    blue_channel = image[2049:3072]
    blue_channel = blue_channel[32, 32]

    # Display one of the channels only :D
    plt.imshow(red_channel, interpolation='nearest')




# print("Number of training data", image_set)
# print("Number of labels data", train_labels.shape)
#
# print(image_set.shape)

# single_image = image_set[0]
#
# # print(image_set[0].shape)
# plt.imshow(single_image, interpolation='nearest')
# plt.show()

# Allocate 0.2 of the data as test data
# train_data, test_data, train_labels, test_labels = train_test_split(data, random_state=True, test_size=0.2)

# Create an NN object
# clf = NN()
