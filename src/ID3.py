import numpy as np
from collections import Counter

class ID3:
    def __init__(self):
        self.tree = None
        self.discretization_thresholds = {}
        self.label_encoder = {}
        self.label_decoder = {}

    def _encode_labels(self, labels):
        unique_labels = np.unique(labels)
        
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.label_decoder = {idx: label for label, idx in self.label_encoder.items()}
        
        return np.array([self.label_encoder[label] for label in labels])

    def discretize_continuous_attribute(self, examples, A):
        sorted_indices = np.argsort(examples[:, A])
        X_sorted = examples[sorted_indices]
        y_sorted = examples[:, -1][sorted_indices]
        
        breakpoints = []
        for i in range(1, len(X_sorted)):
            if y_sorted[i] != y_sorted[i - 1]:
                midpoint = (X_sorted[i, A] + X_sorted[i - 1, A]) / 2
                breakpoints.append(midpoint)
        
        best_gain = -1
        best_threshold = None
        for threshold in breakpoints:
            left_mask = X_sorted[:, A] < threshold
            right_mask = ~left_mask
            
            y_left = y_sorted[left_mask]
            y_right = y_sorted[right_mask]
            
            left_examples = X_sorted[left_mask]
            right_examples = X_sorted[right_mask]
            left_full = np.column_stack([left_examples, y_left])
            right_full = np.column_stack([right_examples, y_right])
            
            gain = self.information_gain_for_split(y_sorted, left_full, right_full)
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
        
        return best_threshold

    def information_gain_for_split(self, y_original, left_examples, right_examples):
        total_len = len(y_original)
        left_len = len(left_examples)
        right_len = len(right_examples)
        
        original_entropy = self.entropy_from_labels(y_original)
        left_entropy = self.entropy_from_labels(left_examples[:, -1])
        right_entropy = self.entropy_from_labels(right_examples[:, -1])
        
        weighted_split_entropy = (
            (left_len / total_len) * left_entropy + 
            (right_len / total_len) * right_entropy
        )
        
        return original_entropy - weighted_split_entropy

    def entropy_from_labels(self, labels):
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        return np.sum([-p * np.log2(p) for p in probabilities])

    def plurality_value(self, examples):
        most_common_encoded = Counter(examples[:, -1]).most_common(1)[0][0]
        return self.label_decoder[most_common_encoded]

    def check_all_example_same(self, examples):
        return np.all(examples[:, -1] == examples[0, -1])

    def entropy(self, S):
        y = S[:, -1]
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return np.sum([-p * np.log2(p) for p in probabilities])

    def information_gain(self, S, A):
        # Hitung Entropy(S)
        entropy_S = self.entropy(S)

        # Hitung information gain
        Xa = S[:, A]
        values_A, len_Sv = np.unique(Xa, return_counts=True)
        len_S = len(S)

        return entropy_S - sum(
            (len_Sv[i] / len_S) * self.entropy(S[Xa == v]) for i, v in enumerate(values_A)
        )
    
    def split_information(self, S, A):
        Xa = S[:, A]
        Si_values, len_Si = np.unique(Xa, return_counts=True)
        len_S = len(S)
        ratio_Si_S = len_Si / len_S
        return -np.sum(ratio * np.log2(ratio) for ratio in ratio_Si_S)
    
    def gain_ratio(self, S, A):
        return self.information_gain(S, A) / self.split_information(S, A)

    def importance(self, a, examples, metric):
        if metric == "information_gain":
            return self.information_gain(examples, a)
        elif metric == "gain_ratio":
            return self.gain_ratio(examples, a)
        return -1
    
    def argmax(self, examples, attributes):
        best_value = -1
        best_attribute = None
        for a in attributes:
            metric = "information_gain"
            if a in self.continuous_attributes:
                if a not in self.discretization_thresholds:
                    self.discretization_thresholds[a] = self.discretize_continuous_attribute(examples, a)
                
                threshold = self.discretization_thresholds[a]
                binary_examples = examples.copy()
                binary_examples[:, a] = (binary_examples[:, a] < threshold).astype(int)
                value = self.importance(a, binary_examples, metric)
            else:
                value = self.importance(a, examples, metric)
            
            if value > best_value:
                best_value = value
                best_attribute = a
        return best_attribute

    def decision_tree_learning(self, examples, attributes, parent_examples):
        if len(examples) == 0:
            return self.plurality_value(parent_examples)
        elif self.check_all_example_same(examples):
            return self.label_decoder[examples[0, -1]]
        elif len(attributes) == 0:
            return self.plurality_value(examples)
        else:
            A = self.argmax(examples, attributes)
            tree = {A: {}}
            
            if A in self.continuous_attributes:
                threshold = self.discretization_thresholds[A]
                XA = (examples[:, A] < threshold).astype(int)
            else:
                XA = examples[:, A]
            
            for vk in np.unique(XA):
                if A in self.continuous_attributes:
                    exs = examples[XA == vk]
                else:
                    exs = examples[XA == vk]
                
                subtree = self.decision_tree_learning(exs, [a for a in attributes if a != A], examples)
                tree[A][vk] = subtree
        return tree
    
    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        encoded_labels = self._encode_labels(y_train)
        
        self.continuous_attributes = [
            i for i in range(X_train.shape[1]) 
            if np.issubdtype(X_train[:, i].dtype, np.number)
        ]
        
        self.discretization_thresholds = {}
        
        examples = np.column_stack([X_train, encoded_labels])
        
        for attr in self.continuous_attributes:
            self.discretization_thresholds[attr] = self.discretize_continuous_attribute(examples, attr)
        
        self.default_class = Counter(encoded_labels).most_common(1)[0][0]
        
        attributes = list(range(X_train.shape[1]))
        
        self.tree = self.decision_tree_learning(examples, attributes, examples)
        return self
 
    def predict_single(self, X, tree=None):
        if tree is None:
            tree = self.tree
        
        # Base case: Leaf
        if not isinstance(tree, dict):
            return tree
        
        attribute = next(iter(tree))
        
        if attribute in self.continuous_attributes:
            threshold = self.discretization_thresholds[attribute]
            value = int(X[attribute] < threshold)
        else:
            value = X[attribute]
        
        if value not in tree[attribute]:
            return self.label_decoder[self.default_class]
        
        return self.predict_single(X, tree[attribute][value])
    
    def predict(self, X):
        X = np.array(X)
        predictions = [self.predict_single(x) for x in X]
        return np.array(predictions)

if __name__ == "__main__":
    X_train1 = np.array([
        [2.5, 3.5, 1.2],
        [1.0, 2.2, 3.3],
        [3.3, 3.1, 0.8],
        [0.5, 1.5, 2.5],
        [2.7, 3.0, 1.5],
        [1.2, 2.8, 2.7],
        [3.0, 3.2, 0.5],
        [0.7, 1.8, 2.9]
    ])

    y_train1 = np.array([0, 1, 0, 1, 0, 1, 0, 1])  # Binary labels

    X_test1 = np.array([
        [2.8, 3.2, 1.3],
        [1.1, 2.0, 2.8],
        [3.2, 3.3, 0.7],
        [0.6, 1.6, 2.6]
    ])

    # Initialize and train the classifier
    clf1 = ID3()
    clf1.fit(X_train1, y_train1)

    # Predict on test data
    predictions1 = clf1.predict(X_test1)

    # Print the tree and predictions
    print("Decision Tree 1:")
    print(clf1.tree)

    print("\nPredictions 1:")
    print(predictions1)

    # Sample dataset 2
    X_train2 = np.array([
        [5.1, 3.5, 1.4],
        [4.9, 3.0, 1.4],
        [6.2, 3.4, 5.4],
        [5.9, 3.0, 5.1],
        [5.7, 2.8, 4.1],
        [4.6, 3.6, 1.0],
        [5.0, 3.4, 1.6],
        [6.7, 3.1, 4.7]
    ])

    y_train2 = np.array(['0', '0', '1', '1', '1', '0', '0', '1'])  # Binary labels

    X_test2 = np.array([
        [5.8, 3.1, 4.2],
        [4.8, 3.4, 1.3],
        [6.4, 3.2, 5.3],
        [5.1, 3.8, 1.9]
    ])

    clf2 = ID3()
    clf2.fit(X_train2, y_train2)

    predictions2 = clf2.predict(X_test2)

    print("\nDecision Tree 2:")
    print(clf2.tree)

    print("\nPredictions 2:")
    print(predictions2)

    # Sample dataset 3
    X_train3 = np.array([
        [10.0, 15.0, 1.5],
        [9.5, 14.0, 2.0],
        [12.0, 18.0, 0.5],
        [8.0, 13.0, 2.5],
        [11.0, 16.0, 1.0],
        [9.0, 14.5, 2.2],
        [12.5, 17.5, 0.8],
        [8.5, 13.5, 2.7]
    ])

    y_train3 = np.array(['1', '1', '0', '1', '0', '1', '0', '1'])  # Binary labels

    X_test3 = np.array([
        [10.5, 15.5, 1.8],
        [9.2, 14.2, 2.3],
        [12.3, 17.8, 0.7],
        [8.7, 13.7, 2.6]
    ])

    clf3 = ID3()
    clf3.fit(X_train3, y_train3)

    predictions3 = clf3.predict(X_test3)

    print("\nDecision Tree 3:")
    print(clf3.tree)

    print("\nPredictions 3:")
    print(predictions3)

    # Sample dataset 4
    X_train4 = np.array([
        [0.5, 1.5, 0.8],
        [0.7, 1.8, 1.2],
        [1.2, 2.5, 0.5],
        [0.9, 2.0, 1.0],
        [0.6, 1.7, 0.7],
        [1.1, 2.4, 1.5],
        [0.8, 1.6, 0.9],
        [1.0, 2.2, 1.1]
    ])

    y_train4 = np.array(['0', '1', '0', '1', '0', '1', '0', '1'])  # Binary labels

    X_test4 = np.array([
        [0.6, 1.6, 0.8],
        [1.0, 2.3, 1.3],
        [1.1, 2.6, 0.6],
        [0.7, 1.7, 1.0]
    ])

    clf4 = ID3()
    clf4.fit(X_train4, y_train4)

    predictions4 = clf4.predict(X_test4)

    print("\nDecision Tree 4:")
    print(clf4.tree)

    print("\nPredictions 4:")
    print(predictions4)
