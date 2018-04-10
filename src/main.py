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
from knn import NearestNeighbour
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

# Create the Nearest Neighbour model
model = NearestNeighbour()

for index, batch in enumerate(BATCH_LIST):
    data, labels = load_cifar(DATA_ROOT + batch)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2)
    model.train(X_train, y_train)
    print("Batch %d: " %(index))
    print(model.predict(X_test))
    

# Load the first batch only
# train_data, labels = load_cifar(DATA_ROOT + BATCH_LIST[0])
# image = train_data[2]

# red_channel = image[0:1024]
# red_channel = red_channel.reshape(32, 32)
# green_channel = image[1024: 2048]
# green_channel = green_channel.reshape(32, 32)
# blue_channel = image[2048:3072]
# blue_channel = blue_channel.reshape(32, 32)

# # Combine the channels

# # Display one of the channels only :D
# plt.imshow(blue_channel, interpolation='nearest')
# plt.show()
# imr=Image.fromarray(red_channel) # mode I
# imb=Image.fromarray(blue_channel)
# img=Image.fromarray(green_channel)

# merged=Image.merge("RGB",(imr,img,imb))
# merged.show()



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
