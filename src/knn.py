# ========================================
# [] File Name : knn.py
#
# [] Creation Date : April 2018
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================


import numpy as np


class NearestNeighbour:
    def __init__(self):
        self.train_data = np.array([])
        self.train_labels = np.array([])
        self.predicted_labels = np.array([])

    def train(self, train_data, train_labels):
        # There not much training in KNN
        self.train_data = train_data
        self.train_labels = train_labels

    def predict(self, test_data):
        # Loop over all rows test data
        for row in test_data:
            distances = np.sum(np.subtract(test_data[row], self.train_data[0, :]))

            # Select the minimum distance as the prediction
            self.predicted_labels = np.append(self.predicted_labels, np.argmin(distances))

        # Return the predicted labels
        return self.predicted_labels
