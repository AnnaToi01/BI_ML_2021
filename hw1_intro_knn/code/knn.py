import numpy as np


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """

    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict classes for the data samples provided

        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """

        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loop(X)
        else:
            distances = self.compute_distances_two_loops(X)

        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)

    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run

        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        distances = np.zeros((X.shape[0], self.train_X.shape[0]))
        for i, v_test in enumerate(X):
            for j, v_train in enumerate(self.train_X):
                distances[i, j] = np.sum(np.abs(v_test - v_train))
        return distances

    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so that only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run

        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        distances = np.zeros((X.shape[0], self.train_X.shape[0]))
        for i, v_test in enumerate(X):
            distances[i, :] = np.sum(np.abs(v_test - self.train_X), axis=1)
        return distances

    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run

        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        """
        YOUR CODE IS HERE
        """
        distances = np.zeros((X.shape[0], self.train_X.shape[0]))
        distances = np.sum(np.abs(X[:, np.newaxis, :] - self.train_X[np.newaxis, ...]), axis=2)
        return distances

    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case

        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions
           for every test sample
        """

        # n_train = distances.shape[1]
        # n_test = distances.shape[0]
        # print(distances.shape)
        # print(lst_idx.shape)
        if self.k == 1:
            prediction = self.train_y[np.argmin(distances, axis=1)]
        else:
            n_test = distances.shape[0]
            prediction = np.zeros(n_test)
            lst_idx = np.argsort(distances, axis=1)[:, :self.k]
            # print(lst_idx)
            for i, idx in enumerate(lst_idx):
                neighbor_classes = self.train_y[np.array(idx)]
                # print(neighbor_classes)
                values, counts = np.unique(neighbor_classes, return_counts=True)
                prediction[i] = np.random.choice(
                    values[counts == counts.max()])  # Random chose when both same frequency

                # prediction[i] = np.argmax(np.bincount(neighbor_classes))
                # print(prediction[i])
        return prediction

    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case

        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index
           for every test sample
        """

        # n_train = distances.shape[0]
        # n_test = distances.shape[0]
        # prediction = np.zeros(n_test, np.int)
        if self.k == 1:
            prediction = self.train_y[np.argmin(distances, axis=1)]
        else:
            n_test = distances.shape[0]
            prediction = np.zeros(n_test)
            lst_idx = np.argsort(distances, axis=1)[:, :self.k]
            # print(lst_idx)
            for i, idx in enumerate(lst_idx):
                neighbor_classes = self.train_y[np.array(idx)]
                # print(neighbor_classes)
                values, counts = np.unique(neighbor_classes, return_counts=True)
                prediction[i] = np.random.choice(
                    values[counts == counts.max()])  # Random chose when both same frequency

                # prediction[i] = np.argmax(np.bincount(neighbor_classes))
                # print(prediction[i])
        return prediction
