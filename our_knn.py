import numpy as np
import numba as nb


class KNNAnomalyDetector:
    def __init__(self, k=5, threshold_percentile=95, distance_func=None, score_func=None, external_threshold=None):
        self.k = k
        self.threshold_percentile = threshold_percentile
        self.distance_func = distance_func if distance_func is not None else self.default_distance
        self.score_func = score_func if score_func is not None else self.default_score
        self.external_threshold = external_threshold
        self.threshold_ = None
        self.anomaly_scores_ = None

    def fit(self, X):
        X = np.array(X)
        self.n_samples = len(X)
        self.X_train = X.copy()  # save train data copy
        
        distances = self.calc_distances(self.n_samples, self.X_train, self.distance_func)

        self.anomaly_scores_ = self.score_func(distances, self.X_train, self.k)

        # if there's no threshold given, we determine treshold using percentile
        if self.external_threshold is None:
            self.threshold_ = np.percentile(self.anomaly_scores_, self.threshold_percentile)
        else:
            self.threshold_ = self.external_threshold
        return self
        
    def predict(self, X, threshold=None):
        # use external threshold if given, otherwise self.threshold_
        X = np.array(X)
        if threshold is None:
            threshold = self.threshold_
        scores = self.decision_function(X)
        predictions = (scores > threshold).astype(int)
        # scale output to match sklearn anomaly detection functions
        return predictions * -2 + 1 
    
    def fit_predict(self, X, threshold=None):
        self.fit(X)
        return self.predict(X, threshold=threshold)
    
    def decision_function(self, X):
        # proess input data instead of train data
        X = np.array(X)
        
        distances = self.calc_distances_pred(self.X_train, X, self.distance_func)

        scores = self.score_func(distances, X, self.k)

        return scores
    
    @staticmethod
    def default_score(distances, X, k):
        # default decition is based on mean distance to k nearest neighbors
        nearest_neighbors = np.argsort(distances, axis=1)[:, :k]
        return np.mean(distances[np.arange(len(X))[:, None], nearest_neighbors], axis=1)

    @staticmethod
    @nb.jit(nopython=True, parallel=True) 
    def calc_distances(n_samples, X, distance_func):
        # compute distance between points
        distances = np.zeros((n_samples, n_samples))
        for i in nb.prange(n_samples):
            for j in range(i + 1, n_samples):
                distances[i, j] = distance_func(X[i], X[j])
                distances[j, i] = distances[i, j]
        return distances
    
    @staticmethod
    @nb.jit(nopython=True, parallel=True) 
    def calc_distances_pred(X_train, X, distance_func):
        distances = np.zeros((len(X), len(X_train)))
        for i in nb.prange(len(X)):
            for j in range(len(X_train)):
                distances[i, j] = distance_func(X[i], X_train[j])
        return distances

    @staticmethod
    @nb.jit(nopython=True)  
    def default_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    @staticmethod
    @nb.jit(nopython=True)  
    def kth_neighbours_distance(self, distances, k):
        nearest_neighbors = np.argsort(distances, axis=1)
        kth_distances = distances[np.arange(distances.shape[0]), nearest_neighbors[:, k]]
        return kth_distances