__authors__ = ['1666134','1703200','1668438','1672891']
__group__ = '07'

import numpy as np
import utils
from collections import Counter
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

class  KMeans:

    def __init__(self, X, K=1, options=None):
        """
        Constructor of KMeans class
            Args:
                K (int): Number of cluster
                options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self.p = 0.2
        self._init_options(options)  # DICT options
        self.val = 0

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################


    def _init_X(self, X):
        """
            Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        X = np.asarray(X, dtype=float)
        if X.ndim > 2:
            X = X.reshape(-1, X.shape[-1])
        if X.shape[-1] == 3:
            X = X.reshape(-1, 3)
        self.X = X
        
        
    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0.00001
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.
        

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        self.centroids = np.zeros((self.K, self.X.shape[1]))
        self.old_centroids = np.zeros((self.K, self.X.shape[1]))

        if self.options['km_init'].lower() == 'first':
            selected_points = []
            count = 0
            while(len(selected_points) < self.K):
                #si el punt actual no ha estat seleccionat, l'afegim a la llista
                if not any((self.X[count] == p).all() for p in selected_points):
                    selected_points.append(self.X[count])
                count += 1
            self.centroids[:len(selected_points)] = selected_points
            #assignem els punts seleccionats com centroides actuals
        
        elif self.options['km_init'].lower() == 'random':
            # Inicialització aleatòria de centroides
            self.centroids = np.random.rand(self.K, self.X.shape[1])
            
        elif self.options['km_init'].lower() == 'kmeans++':
            self.centroids[0] = self.X[np.random.randint(self.X.shape[0])]  # Seleccionar el primer centroide aleatoriament
            
            for i in range(1, self.K):
                # Calcular les distàncies al quadrat entre els punts de dades i els centroides ms propers seleccionats
                distances = distance(self.X, self.centroids[:i])**2
                
                min_distances = distances.min(axis=1)
                
                probabilities = min_distances / min_distances.sum()
                # Seleccionar un nou centroide aleatoriament basat en les probabilitats
                self.centroids[i] = self.X[np.random.choice(range(self.X.shape[0]), p=probabilities)]
        
        


    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        closest_centroids_indices = np.argmin(distance(self.X, self.centroids), axis=1)
        self.labels = closest_centroids_indices


    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        self.old_centroids = self.centroids.copy()
        for k in range(self.K):
            cluster_points = self.X[self.labels == k] 
            if len(cluster_points) > 0:
                self.centroids[k] = np.mean(cluster_points, axis=0)


    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        centroid_diff = np.abs(self.centroids - self.old_centroids)
        total_sum = np.sum(centroid_diff)
        return total_sum < self.options['tolerance']
    

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        self.num_iter = 0
        self._init_centroids()
        while(self.converges() == False and self.num_iter < self.options['max_iter']):
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1
    
    
    def withinClassDistance(self):
        """
        returns the within class distance of the current clustering
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        dist = np.linalg.norm(self.X - self.centroids[self.labels], axis = 1)
        return np.mean(dist * np.transpose(dist))
        
    def inter_class(self):
        inter = 0
        for i in range(self.K):
            for j in range(self.K):
                if i!=j:
                    C1 = self.X[self.labels == i]
                    C2 = self.X[self.labels == j]
                    inter += np.mean(np.power(C1[:,np.newaxis]-C2,2))

        return inter
    
    def fisher_disc(self):
        intra_class=self.withinClassDistance()
        inter_class=self.inter_class()  
        return intra_class/inter_class


    def find_bestK(self, max_K):
        """
        sets the best k analysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################                
                
        prev = float('inf')  
        for K in range(2, max_K+1):
            self.K = K
            self.fit()
            if self.options['fitting'] == 'WCD':
                now = self.withinClassDistance()
            elif self.options['fitting']=='IC':
                now = self.inter_class()
            elif self.options['fitting'] == 'FD': 
                now = self.fisher_disc()
            if K > 2:
                if self.options['fitting']=='IC': 
                    if 1 - prev/now > self.p: 
                        self.K = K - 1 
                        self.val = now
                        break
                else:
                    if 1 - now/prev < self.p:
                        self.K = K - 1
                        self.val = now
                        break 
            prev = now
                

            
def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    ######################################################### 

    return np.linalg.norm(X[:, np.newaxis, :] - C, axis=2)


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """
    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    
    return utils.colors[np.argmax(utils.get_color_prob(centroids), axis = 1)]
