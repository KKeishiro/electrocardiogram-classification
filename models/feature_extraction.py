import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import ndimage
from biosppy.signals import ecg


def calcFeatures(segment, n_bins):
    features = []
    assert(len(segment.shape) == 3)
    total_hist = np.histogram(segment, bins=n_bins, range=(240, 1800))[0]
    features.extend(total_hist.tolist())
    return features


class Testsampling(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_new = X[0:10]
        return X_new


class Trimming(BaseEstimator, TransformerMixin):
    """Trimming the beginning and the end of the data"""
    def __init__(self, start=500, end=7500):
        self.start = start
        self.end = end

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("trimming started!")
        X_new = X[:, self.start:self.end]
        print("trimming is done!")
        return X_new


class ExtractHeartbeat(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("transforming started!")
        X_new = []
        for row in X:
            templates = ecg.ecg(
                        row, sampling_rate=300.0, show=False)['templates']
            features = np.median(templates, axis=0)
            X_new.append(features)
        X_new = np.array(X_new)
        print("shape before ...", X.shape, "shape after ...", X_new.shape)
        return X_new


class GetRRInterval(BaseEstimator, TransformerMixin):
    """Get RR intervals of the samples"""
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("transforming started!")
        X_rr_mean = []
        for row in X:
            rpeaks = ecg.ecg(row, sampling_rate=300.0, show=False)['rpeaks']
            rr = np.diff(rpeaks)
            assert len(rr) != 0
            X_rr_mean.append(np.mean(rr))
        X_rr_mean = np.array(X_rr_mean)
        X_rr_mean = X_rr_mean.reshape(-1, 1)
        return X_rr_mean


class CutOff(BaseEstimator, TransformerMixin):
    """Cuts off the irrelevant black areas."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.reshape(X, (-1, 176, 208, 176))
        X = X[:, 20:150, 20:180, 20:150]

        return X.astype(np.float64)


class SegmentFeatures(BaseEstimator, TransformerMixin):
    """Extracts histograms images"""

    def __init__(self, n_bins=100):
        self.n_bins = n_bins

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = []
        print("Shape before", X.shape)
        n_samples = X.shape[0]
        for i in range(n_samples):
            features = []
            for x in range(0, 10):
                for y in range(0, 10):
                    for z in range(0, 10):
                        segment = X[i, (x*13):((x+1)*13),
                                       (y*16):((y+1)*16),
                                       (z*13):((z+1)*13)]
                        features.extend(calcFeatures(segment, self.n_bins))
            for x in range(X.shape[3]):
                hist = np.histogram(
                          ndimage.gaussian_filter(X[i, :, :, x], sigma=2.5),
                          bins=100, range=(100, 1400))[0]
                hist = np.histogram(ndimage.prewitt(X[i, :, x, :]),
                                    bins=100, range=(240, 2400))[0]
                features.extend(hist.tolist())
            for x in range(X.shape[2]):
                hist = np.histogram(
                        ndimage.gaussian_filter(X[i, :, x, :], sigma=2.5),
                        bins=100, range=(100, 1400))[0]
                hist = np.histogram(ndimage.prewitt(X[i, :, x, :]),
                                    bins=100, range=(240, 2400))[0]
                features.extend(hist.tolist())
            for x in range(X.shape[1]):
                hist = np.histogram(
                        ndimage.gaussian_filter(X[i, x, :, :], sigma=2.5),
                        bins=100, range=(100, 1400))[0]
                hist = np.histogram(ndimage.prewitt(X[i, x, :, :]),
                                    bins=100, range=(240, 2400))[0]
                features.extend(hist.tolist())
            X_new.append(features)

        X_new = np.asarray(X_new)
        print(X_new.shape)
        return X_new
