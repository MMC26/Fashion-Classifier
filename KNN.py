__authors__ = ['1666134','1703200','1668438','1672891']
__group__ = '07'

import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter

class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.test_data = np.array
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.train_data = np.asarray(train_data, dtype=float)
        self.train_data = train_data.reshape(train_data.shape[0], -1)


    def get_k_neighbours(self, test_data, k, distance_type='euclidean'):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :param distance_type: type of distance to use (euclidean, manhattan, chebyshev, cosine)
        :return: the matrix self.neighbors is created (NxK)
                the ij-th entry is the j-th nearest train point to the i-th test point
        """
        self.test_data = np.asarray(test_data, dtype=float)
        P,M,N,D = test_data.shape
        self.test_data = self.test_data.reshape(P, M*N*D)
        
        if distance_type == 'euclidean':
            distances = cdist(self.test_data, self.train_data, 'euclidean')
        elif distance_type == 'manhattan':
            distances = cdist(self.test_data, self.train_data, 'cityblock')
        elif distance_type == 'chebyshev':
            distances = cdist(self.test_data, self.train_data, 'chebyshev')
        elif distance_type == 'cosine':
            distances = cdist(self.test_data, self.train_data, 'cosine')
        elif distance_type == 'minkowski':
            distances = cdist(self.test_data, self.train_data, 'minkowski')
        elif distance_type == 'canberra':
            distances = cdist(self.test_data, self.train_data, 'canberra')
        elif distance_type == 'hamming':
            distances = cdist(self.test_data, self.train_data, 'hamming')
        else:
            raise ValueError("Invalid distance type. Supported types are 'euclidean', 'manhattan', 'chebyshev', 'cosine'.")
        
        self.neighbors = self.labels[np.argsort(distances, axis=1)[:, :k]]
        

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        indexes = []
        for neighbor in self.neighbors:
            counter=Counter(neighbor).most_common(1)[0][0]
            indexes.append(counter)
        
        return indexes

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
