import numpy as np

class KNNAnomalyDetector:
    def __init__(self, k=5, threshold_percentile=95, distance_func=None, external_threshold=None):
        self.k = k
        self.threshold_percentile = threshold_percentile
        self.distance_func = distance_func if distance_func is not None else self.default_distance
        self.external_threshold = external_threshold
        self.threshold_ = None
        self.anomaly_scores_ = None

    def fit(self, X):
        X = np.array(X)

        self.n_samples = len(X)
        self.X_train = X.copy()  # save train data copy
        distances = np.zeros((self.n_samples, self.n_samples))

        # compute distance between points
        for i in range(self.n_samples):
            for j in range(i + 1, self.n_samples):
                distances[i, j] = self.distance_func(X[i], X[j])
                distances[j, i] = distances[i, j]

        # find k nearest neighbors
        nearest_neighbors = np.argsort(distances, axis=1)[:, 1:self.k]
        # mean distance from k nearest neighbors
        self.anomaly_scores_ = np.mean(distances[np.arange(self.n_samples)[:, None], nearest_neighbors], axis=1)

        # if there's no threshold given, we determine threshold using percentile
        if self.external_threshold is None:
            self.threshold_ = np.percentile(self.anomaly_scores_, self.threshold_percentile)
        else:
            self.threshold_ = self.external_threshold
        return self

    def predict(self, X, threshold=None):
        X = np.array(X)
        # use external threshold if given, otherwise self.threshold_
        if threshold is None:
            threshold = self.threshold_
        scores = self.decision_function(X)
        predictions = (scores > threshold).astype(int)
        # scale output to match sklearn anomaly detection functions
        return predictions * -2 + 1 
    
    def fit_predict(self, X, threshold=None):
        self.fit(X)
        return self.predict(X, threshold)

    def decision_function(self, X):
        X = np.array(X)
        # proses input data instead of train data
        distances = np.zeros((len(X), len(self.X_train)))
        for i in range(len(X)):
            for j in range(len(self.X_train)):
                distances[i, j] = self.distance_func(X[i], self.X_train[j])

        nearest_neighbors = np.argsort(distances, axis=1)[:, :self.k]
        scores = np.mean(distances[np.arange(len(X))[:, None], nearest_neighbors], axis=1)
        return scores

    @staticmethod
    def default_distance(x1, x2):
        # Euclidean distance
        return np.sqrt(np.sum((x1 - x2) ** 2))