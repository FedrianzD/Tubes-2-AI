from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
class FeatureScaler(BaseEstimator, TransformerMixin):

    def __init__(self, feature_range=(0,1)):
        self.feature_range = feature_range

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.min = X.min(axis=0).values
        self.max = X.max(axis=0).values
        range_min, range_max = self.feature_range
        self.scale = (range_max - range_min) / (self.max - self.min)
        return self

    def transform(self, X):
        X_scaled = X.copy()
        X_scaled = pd.DataFrame(X_scaled)
        range_min, range_max = self.feature_range
        X_scaled = (X_scaled - self.min) * self.scale + range_min
        return X_scaled
    