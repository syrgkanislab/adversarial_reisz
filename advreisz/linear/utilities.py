from sklearn.base import BaseEstimator, TransformerMixin
import copy
import numpy as np
from functools import reduce


def cross_product(*XS):
    """
    Compute the cross product of features.

    Parameters
    ----------
    X1 : n x d1 matrix
        First matrix of n samples of d1 features
        (or an n-element vector, which will be treated as an n x 1 matrix)
    X2 : n x d2 matrix
        Second matrix of n samples of d2 features
        (or an n-element vector, which will be treated as an n x 1 matrix)

    Returns
    -------
    A : n x (d1*d2*...) matrix
        Matrix of n samples of d1*d2*... cross product features,
        arranged in form such that each row t of X12 contains:
        [X1[t,0]*X2[t,0]*..., ..., X1[t,d1-1]*X2[t,0]*..., X1[t,0]*X2[t,1]*..., ..., X1[t,d1-1]*X2[t,1]*..., ...]

    """
    for X in XS:
        assert 2 >= np.ndim(X) >= 1
    n = np.shape(XS[0])[0]
    for X in XS:
        assert n == np.shape(X)[0]

    def cross(XS):
        k = len(XS)
        XS = [np.reshape(XS[i], (n,) + (1,) * (k - i - 1) + (-1,) + (1,) * i)
              for i in range(k)]
        return np.reshape(reduce(np.multiply, XS), (n, -1))
    return cross(XS)


class SeparateFeaturizer(TransformerMixin, BaseEstimator):

    def __init__(self, featurizer):
        self.featurizer = copy.deepcopy(featurizer)

    def transform(self, X):
        T = X[:, [0]]
        feats = self.featurizer.transform(X[:, 1:])
        return np.hstack([T * feats, (1 - T) * feats])

    def fit(self, X, y=None):
        self.featurizer.fit(X[:, 1:], y)
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names(self):
        return ['T0 * {}'.format(i) for i in self.featurizer.get_feature_names()] +\
            ['T1 * {}'.format(i) for i in self.featurizer.get_feature_names()]


class CoordinatePolynomialFeatures(TransformerMixin, BaseEstimator):

    def __init__(self, degree=1):
        self.degree = degree

    def transform(self, X):
        for i in np.arange(2, self.degree + 1):
            X = np.hstack([X, X**i])
        return X

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)
