import numpy as np
from sklearn.base import BaseEstimator, clone


class FitParamsWrapper:

    def __init__(self, model, **fit_params):
        self.model = model
        self.fit_params = fit_params
    
    def fit(self, X):
        return self.model.fit(X, **self.fit_params)
    
    def predict(self, X):
        return self.model.predict(X)

class PluginRR(BaseEstimator):

    def __init__(self, *, model_t, min_propensity=0):
        self.model_t = clone(model_t, safe=False)
        self.min_propensity = min_propensity
    
    def fit(self, X):
        self.model_t_ = clone(self.model_t, safe=False).fit(X[:, 1:], X[:, 0])
        return self
    
    def predict(self, X):
        propensity = np.clip(self.model_t_.predict_proba(X[:, 1:]), self.min_propensity, 1 - self.min_propensity)
        return X[:, 0] / propensity[:, 1] - (1 - X[:, 0]) / propensity[:, 0]