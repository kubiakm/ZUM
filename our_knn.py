import numpy as np
import numba as nb
from sklearn.base import BaseEstimator, ClassifierMixin


class KNNAnomalyDetector(ClassifierMixin, BaseEstimator):
    def __init__(self, k=5, threshold_percentile=95, distance_func=None, score_func=None, external_threshold=None):
        distanceLookup = {
            "euclidean": self.default_distance,
            "minkowski3": self.minkowski_distance_p3,
            "manhattan": self.manhattan_distance
        }

        scoreLookup = {
            "avgNNearest": self.default_score,
            "distFromNth": self.distance_from_nth,
            "averageDist": self.distance_avg,
            "density": self.density,
            "LOF": self.lof_score
        }

        self.k = k
        self.threshold_percentile = threshold_percentile
        if distance_func is None:
            self.distance_func = self.default_distance # euclidean
        elif type(distance_func) == str:
            self.distance_func = distanceLookup[distance_func]
        else:
            self.distance_func = distance_func

        if score_func is None:
            self.score_func = self.default_score #avg n neighbors
        elif type(score_func) == str:
            self.score_func = scoreLookup[score_func]
        else:
            self.score_func = score_func

        self.external_threshold = external_threshold
        self.threshold_ = None
        self.anomaly_scores_ = None

        self.classes_ = [-1, 1]

    def fit(self, X):
        X = np.array(X)
        self.n_samples = len(X)
        self.X_train = X.copy()  # save train data copy
        
        distances = self.calc_distances_fit(self.X_train, self.distance_func)

        self.anomaly_scores_ = self.score_func(distances, self.X_train, self.k)

        # if there's no threshold given, we determine treshold using percentile
        if self.external_threshold is None:
            self.threshold_ = np.percentile(self.anomaly_scores_, self.threshold_percentile)
        else:
            self.threshold_ = self.external_threshold
        return self
        
    def predict(self, X, threshold=None):
        X = np.array(X)
        if threshold is None:
            threshold = self.threshold_
        scores = self.decision_function(X)
        predictions = (scores < threshold).astype(int)
        # scale output to match sklearn anomaly detection functions
        return predictions * -2 + 1 
    
    def fit_predict(self, X, threshold=None):
        self.fit(X)
        return self.predict(X, threshold=threshold)
    
    def decision_function(self, X):
        # proses input data instead of train data
        X = np.array(X)
        distances = self.calc_distances(self.X_train, X, self.distance_func)
        scores = self.score_func(distances, X, self.k)

        return scores
    
    @staticmethod
    def default_score(distances, X, k):
        # default decision is based on mean distance to k nearest neighbors
        nearest_neighbors = np.argsort(distances, axis=1)[:, :k]
        nearest_neighbors = nearest_neighbors.reshape((-1, k)) # ensure 2d
        return np.mean(distances[np.arange(len(X))[:, None], nearest_neighbors], axis=1)
    
    @staticmethod
    @nb.jit(nopython=True, parallel=True) 
    def calc_distances_fit(X, distance_func):
        # compute distance between points
        n_samples = len(X)
        distances = np.zeros((n_samples, n_samples))
        for i in nb.prange(n_samples):
            for j in nb.prange(i + 1, n_samples):
                distances[i, j] = distance_func(X[i], X[j])
                distances[j, i] = distances[i, j]
        return distances

    @staticmethod
    @nb.jit(nopython=True, parallel=True) 
    def calc_distances(X_train, X, distance_func):
        distances = np.zeros((len(X), len(X_train)))
        for i in nb.prange(len(X)):
            for j in nb.prange(len(X_train)):
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
    
    def lof_score(self, distances, X, k):
        nearest_neighbors = np.argsort(distances, axis=1)[:, :k]
        distance_from_k = self.distance_from_nth(distances, X, k)
        lofs = np.zeros(len(nearest_neighbors))
        for i, x in enumerate(nearest_neighbors):
            distance_neighbor = self.calc_distances(self.X_train, self.X_train[nearest_neighbors[i]], self.distance_func)
            distances = self.distance_from_nth(distance_neighbor, self.X_train[nearest_neighbors[i]], k)
            lofs[i] = distance_from_k[i] / (np.mean(distances) + 0.01)
        return lofs

    @staticmethod
    def distance_from_nth(distances, X, k):
        kth_nearest_neighbor = np.argsort(distances, axis=1)[:, k-1:k]
        return distances[np.arange(len(X))[:, None], kth_nearest_neighbor]

    @staticmethod
    @nb.jit(nopython=True)
    def manhattan_distance(x1, x2):
        return np.sum(np.abs(x1 - x2))

    @staticmethod
    @nb.jit(nopython=True)
    def minkowski_distance_p3(x1, x2, p=3):
        return np.sum(np.abs(x1 - x2)**p)**(1/p)
    
    @staticmethod
    def distance_from_nth(distances, X, k):
        kth_nearest_neighbor = np.argsort(distances, axis=1)[:, k-1:k]
        return distances[np.arange(len(X))[:, None], kth_nearest_neighbor]

    @staticmethod
    def distance_avg(distances, X, k):
        return np.mean(distances, axis=1)

    @staticmethod
    def density(distances, X, k, fitted_detector=None):
        nearest_neighbors = np.argsort(distances, axis=1)[:, :k]
        if fitted_detector is None:
            fitted_detector = KNNAnomalyDetector(k=k)
            fitted_detector.fit(nearest_neighbors)
        neighbor_scores = fitted_detector.predict(nearest_neighbors)
        neighbor_scores = nearest_neighbors.reshape((-1, k)) # ensure 2d
        return np.mean(neighbor_scores, axis=1)