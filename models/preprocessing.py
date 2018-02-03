import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# X_rr_train = np.load("data/20171214-224732/X_20171214-224732.npy")
X_rr_test = np.load("data/20171215-003826/X_20171215-003826.npy")


class Concatenate(BaseEstimator, TransformerMixin):
    def __init__(self, X2=X_rr_test):
        self.X2 = X2

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_new = np.c_[X, self.X2]
        print("shape before", X.shape, "shape after", X_new.shape)
        return X_new
