import numpy as np

class NearestNeighbour:
    def __init__(self):
        self.train_data = np.array([])
        self.train_labels = np.array([])
        self.predicted_labels = np.array([])

    def train(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels

    def predict(self, test_data):
        # Loop over all rows test data
        for index in test_data:
            distances = np.sum(np.subtract(test_data[index], self.train_data[0, :]))

            # Select the minimum distance as the prediction
            self.predicted_labels[index] = np.argmin(distances)

        # Return the predicted labels
        return self.predicted_labels