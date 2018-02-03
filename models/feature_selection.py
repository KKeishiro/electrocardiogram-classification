from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectPercentile, SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.random import sample_without_replacement


class NonZeroSelection(BaseEstimator, TransformerMixin):
    """Select non-zero voxels"""
    def fit(self, X, y=None):
        X = check_array(X)
        self.nonzero = X.sum(axis=0) > 0
        return self

    def transform(self, X, y=None):
        print("Shape before", X.shape)
        check_is_fitted(self, ["nonzero"])
        X = check_array(X)
        X_new = X[:, self.nonzero]
        print("Shape after", X_new.shape)
        return X_new


class RandomSelection(BaseEstimator, TransformerMixin):
    """Random Selection of features"""
    def __init__(self, n_components=1000, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.components = None

    def fit(self, X, y=None):
        X = check_array(X)
        n_samples, n_features = X.shape

        random_state = check_random_state(self.random_state)
        self.components = sample_without_replacement(
                            n_features,
                            self.n_components,
                            random_state=random_state)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["components"])
        X = check_array(X)
        n_samples, n_features = X.shape
        X_new = X[:, self.components]

        return X_new


class SelectPercentile(SelectPercentile):
    def __init__(self, score_func=f_regression, percentile=60):
        super(SelectPercentile, self).__init__(score_func=score_func)
        self.percentile = percentile


class SelectKBest(SelectKBest):
    def __init__(self, score_func=f_regression, k=1500):
        super(SelectKBest, self).__init__(score_func=score_func)
        self.k = k
