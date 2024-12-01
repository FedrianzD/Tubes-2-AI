import numpy as np
from collections import Counter
class KNN:
    def __init__(self, k, metric):
        self.k = k
        self.metric = metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a-b)**2))
    def manhattan_distance(self, a, b):
        return np.sum(np.abs(a-b))
    def minskowski_distance(self, a, b, p):
        return np.sum(np.abs(a-b)**p)**(1/p)
    
    def predict_single(self, X):
        if self.metric == 'euclidean':
            distances = [self.euclidean_distance(x_train, X) for x_train in self.X_train]
        elif self.metric == 'manhattan':
            distances = [self.manhattan_distance(x_train, X) for x_train in self.X_train]
        elif self.metric == 'minskowski':
            distances = [self.minskowski_distance(x_train, X, 3) for x_train in self.X_train]
        else:
            raise ValueError('Invalid metric')
        
        # sort by distance
        distance_sorted = np.argsort(distances)

        # ambil k elemen
        index_of_k_element = distance_sorted[:self.k]
        k_element_labels = [self.y_train[i] for i in index_of_k_element]
        
        # cari mayoritas label pada k elemen
        most_common = Counter(k_element_labels).most_common(1)
        return most_common[0][0]
    
    def predict(self, X):
        predictions = [self.predict_single(x) for x in X]
        return np.array(predictions)

if __name__ == "__main__":
    X_train = np.array(np.random.randint(1, 10, (5, 2)))
    print("X_train:")
    print(X_train)
    y_train = np.array(np.random.randint(0, 2, 5))
    print("y_train:")
    print(y_train)
    X_test = np.array(np.random.randint(1, 10, (2, 2)))
    print("X_test:")
    print(X_test)
    
    knn = KNN(k=3, metric='manhattan')
    knn.fit(X_train, y_train)
    
    predictions = knn.predict(X_test)
    print(f"Predictions: {predictions}")