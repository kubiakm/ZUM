import numpy as np
class KNNAnomalyDetector:
    def __init__(self, k=5, threshold_percentile=95, distance_func=None, external_threshold=None, knn_or_kth=None):
        self.k = k
        self.threshold_percentile = threshold_percentile
        self.distance_func = distance_func if distance_func is not None else self.default_distance
        self.external_threshold = external_threshold
        self.threshold_ = None
        self.anomaly_scores_ = None
        self.knn_or_kth = knn_or_kth

    def fit(self, X):
        self.n_samples = len(X)
        self.X_train = X.copy()  # save train data copy
        distances = np.zeros((self.n_samples, self.n_samples))

        # compute distance between points
        for i in range(self.n_samples):
            for j in range(i + 1, self.n_samples):
                distances[i, j] = self.distance_func(X[i], X[j])
                distances[j, i] = distances[i, j]

        if self.knn_or_kth == 0:
            self.anomaly_scores_ = self.kth_neighbours_distance(distances)
        else:
            nearest_neighbors = np.argsort(distances, axis=1)[:, 1:self.k+1]
            self.anomaly_scores_ = np.mean(distances[np.arange(self.n_samples)[:, None], nearest_neighbors], axis=1)

        # if there's no threshold given, we determine treshold using percentile
        if self.external_threshold is None:
            self.threshold_ = np.percentile(self.anomaly_scores_, self.threshold_percentile)
        else:
            self.threshold_ = self.external_threshold
        return self
        

    def predict(self, X, threshold=None):
        # use external threshold if given, otherwise self.threshold_
        if threshold is None:
            threshold = self.threshold_
        scores = self.decision_function(X)
        return scores > threshold

    def decision_function(self, X):
        # proess input data instead of train data
        distances = np.zeros((len(X), len(self.X_train)))
        for i in range(len(X)):
            for j in range(len(self.X_train)):
                distances[i, j] = self.distance_func(X[i], self.X_train[j])

        if self.knn_or_kth == 0:
            scores = self.kth_neighbours_distance(distances)
        else:
            nearest_neighbors = np.argsort(distances, axis=1)[:, :self.k]
            scores = np.mean(distances[np.arange(len(X))[:, None], nearest_neighbors], axis=1)
        return scores

    @staticmethod
    def default_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def kth_neighbours_distance(self, distances):
        nearest_neighbors = np.argsort(distances, axis=1)
        kth_distances = distances[np.arange(distances.shape[0]), nearest_neighbors[:, self.k]]
        return kth_distances