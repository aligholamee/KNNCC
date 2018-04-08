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

# Data root path
DATA_ROOT = './data/'
BATCH_LIST = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

# Load the data
train_data = np.array([])
train_labels = np.array([])

for batch in BATCH_LIST:
    data, labels = load_cifar(DATA_ROOT + batch)
    np.append(train_data, data)
    np.append(train_labels, labels)

print(train_data.shape)
print(train_labels.shape)
# Allocate 0.2 of the data as test data
# train_data, test_data, train_labels, test_labels = train_test_split(data, random_state=True, test_size=0.2)

# Create an NN object
# clf = NN()
