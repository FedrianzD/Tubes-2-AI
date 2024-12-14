import numpy as np
from collections import Counter

class ID3:
    def __init__(self):
        self.tree = None

    def discretize_continuous_attribute(self, examples, A):
        sorted_indices = np.argsort(examples[:, A])
        X_sorted = examples[sorted_indices]
        y_sorted = examples[:, -1][sorted_indices]
        
        breakpoints = []
        for i in range(1, len(X_sorted)):
            if y_sorted[i] != y_sorted[i - 1]:
                midpoint = (X_sorted[i, attribute_index] + X_sorted[i - 1, attribute_index]) / 2
                breakpoints.append(midpoint)
        
        best_gain = -1
        best_threshold = None
        for threshold in breakpoints:
            left_mask = X_sorted[:, attribute_index] < threshold
            right_mask = ~left_mask
            
            y_left = y_sorted[left_mask]
            y_right = y_sorted[right_mask]
            
            gain = self.information_gain(y_sorted, y_left, y_right)
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
        
        return best_threshold

    def plurality_value(self, examples):
        return Counter(examples[:, -1]).most_common(1)[0][0]

    def check_all_example_same(self, examples):
        return all(i == list[0] for i in examples[:, -1])

    def entropy(self, S):
        y = S[:, -1]
        label_values, counts = np.unique(y, return_counts=True)
        # counts = np.bincount(y)
        probabilities = counts / len(y)
        return np.sum([-p * np.log2(p) for p in probabilities])

    def information_gain(self, S, A):
        # Hitung Entropy(S)
        entropy_S = self.entropy(S)

        # Hitung Entropy(S) - Sigma bla bla
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
        if (metric == "information_gain"):
            return self.information_gain(examples, a)
        elif (metric == "gain_ratio"):
            return self.gain_ratio(a, )
        return -1
    
    def argmax(self, examples, attributes):

        best_value = -1
        best_attribute = None
        for a in attributes:
            metric = "information_gain" # nanti bisa pilih tergantung avg
            value = self.importance(a, examples, metric)
            if value > best_value:
                best_value = value
                best_attribute = a
        return best_attribute

    def decision_tree_learning(self, examples, attributes, parent_examples):
        if len(examples) == 0:
            return self.plurality_value(parent_examples)
        elif self.check_all_example_same(examples):
            return examples[0, -1]
        elif len(attributes) == 0:
            return self.plurality_value(examples)
        else:
            A = self.argmax(examples, attributes)
            tree = {A: {}}
            XA = examples[:, A]
            for vk in np.unique(XA):
                exs = examples[XA == vk]
                subtree = self.decision_tree_learning(exs, [a for a in attributes if a != A], examples)
                tree[A][vk] = subtree
        return tree
    
    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.default_class = Counter(self.y_train).most_common(1)[0][0]
        examples = np.concatenate((self.X_train, self.y_train.reshape(-1, 1)), axis=1)
        attributes = list(range(self.X_train.shape[1]))
        self.tree = self.decision_tree_learning(examples, attributes, examples)
        return self
 
    def predict_single(self, X, tree):
        # base case: Leaf
        if not isinstance(tree, dict):
            return tree
        
        attribute = next(iter(tree))
        value = X[attribute]
        if value not in tree[attribute]:
            return self.default_class  # kalau value attribute gaada di training
        return self.predict_single(X, tree[attribute][value])
    
    def predict(self, X):
        predictions = [self.predict_single(x, self.tree) for x in X]
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
    
    id3 = ID3()
    id3.fit(X_train, y_train)
    
    predictions = id3.predict(X_test)
    print(f"Predictions: {predictions}")
    print(id3.tree)