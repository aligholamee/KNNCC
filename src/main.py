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

# Data root path
DATA_ROOT = './data/'

# Load the data
data, labels = load_cifar(DATA_ROOT + 'data_batch_1')

print(data.shape)
print(labels.shape)
# Allocate 0.2 of the data as test data
# train_data, test_data, train_labels, test_labels = train_test_split(data, random_state=True, test_size=0.2)

# Create an NN object
# clf = NN()
