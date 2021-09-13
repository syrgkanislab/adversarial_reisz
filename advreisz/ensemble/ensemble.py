# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from scipy.stats.stats import moment
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import BaseEstimator, clone
from econml.grf._base_grf import BaseGRF
from econml.utilities import cross_product

def _mysign(x):
    return 2 * (x >= 0) - 1


def poly_feature_fns(degree):
    def poly(d, sign=1.0):
        return lambda x: sign * x[:, [0]]**d
    return [poly(t) for t in np.arange(0, degree + 1)]


class RFrr(BaseGRF):

    def __init__(self, *,
                 riesz_feature_fns,
                 moment_fn,
                 l2=0.01,
                 n_estimators=100,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=10,
                 min_samples_leaf=5,
                 min_weight_fraction_leaf=0.,
                 min_var_fraction_leaf=None,
                 min_var_leaf_on_val=False,
                 max_features="auto",
                 min_impurity_decrease=0.,
                 max_samples=.45,
                 min_balancedness_tol=.45,
                 honest=True,
                 inference=True,
                 subforest_size=4,
                 n_jobs=-1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        self.riesz_feature_fns = riesz_feature_fns
        self.moment_fn = moment_fn
        self.l2 = l2
        super().__init__(n_estimators=n_estimators,
                         criterion=criterion,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         min_var_fraction_leaf=min_var_fraction_leaf,
                         min_var_leaf_on_val=min_var_leaf_on_val,
                         max_features=max_features,
                         min_impurity_decrease=min_impurity_decrease,
                         max_samples=max_samples,
                         min_balancedness_tol=min_balancedness_tol,
                         honest=honest,
                         inference=inference,
                         fit_intercept=False,
                         subforest_size=subforest_size,
                         n_jobs=n_jobs,
                         random_state=random_state,
                         verbose=verbose,
                         warm_start=warm_start)

    def _get_alpha_and_pointJ(self, X, T, y):
        n_riesz_feats = len(self.riesz_feature_fns)
        TX = np.hstack([T, X])
        riesz_feats = np.hstack([feat_fn(TX)
                                 for feat_fn in self.riesz_feature_fns])
        mfeats = np.hstack([self.moment_fn(TX, feat_fn)
                            for feat_fn in self.riesz_feature_fns])
        alpha = mfeats - y.reshape(-1, 1) * riesz_feats
        riesz_cov_matrix = cross_product(riesz_feats, riesz_feats).reshape(
            (X.shape[0], n_riesz_feats, n_riesz_feats)) + self.l2 * np.eye(n_riesz_feats)
        pointJ = 2 * riesz_cov_matrix
        return alpha, pointJ.reshape((X.shape[0], -1))

    def _get_n_outputs_decomposition(self, X, T, y):
        n_relevant_outputs = len(self.riesz_feature_fns)
        n_outputs = n_relevant_outputs
        return n_outputs, n_relevant_outputs

    def _translate(self, point, TX):
        riesz_feats = np.hstack([feat_fn(TX)
                                 for feat_fn in self.riesz_feature_fns])
        n_riesz_feats = riesz_feats.shape[1]
        riesz = np.sum(point[:, :n_riesz_feats] * riesz_feats, axis=1)
        return riesz

    def predict(self, TX_test):
        point = super().predict(TX_test[:, 1:], interval=False)
        return self._translate(point, TX_test)


class AdvEnsembleReisz(BaseEstimator):

    def __init__(self, *, moment_fn, adversary='auto', learner='auto',
                 max_abs_value=4, n_iter=100, degree=2):
        self.moment_fn = moment_fn
        self.adversary = adversary
        self.learner = learner
        self.max_abs_value = max_abs_value
        self.n_iter = n_iter
        self.degree = degree

    def _get_new_adversary(self):
        return RFrr(riesz_feature_fns=poly_feature_fns(self.degree), moment_fn=self.moment_fn, n_estimators=40, max_depth=2,
                    min_samples_leaf=20, min_impurity_decrease=0.001, inference=False, honest=False) if self.adversary == 'auto' else clone(self.adversary)

    def _get_new_learner(self):
        return RandomForestClassifier(n_estimators=5, max_depth=2, criterion='gini',
                                      bootstrap=False, min_samples_leaf=20, min_impurity_decrease=0.001) if self.learner == 'auto' else clone(self.learner)

    def fit(self, X):
        T, W = X[:, 0], X[:, 1:]
        max_value = self.max_abs_value
        adversary = self._get_new_adversary().fit(W, T, np.zeros(T.shape[0]))
        learners = []
        h = 0
        for it in range(self.n_iter):
            test = adversary.predict(X).flatten()
            aug_T = np.vstack([np.zeros((2, X.shape[1])), X])
            aug_label = np.concatenate(([-1, 1], _mysign(test)))
            aug_weights = np.concatenate(([0, 0], np.abs(test)))
            learners.append(self._get_new_learner().fit(
                aug_T, aug_label, sample_weight=aug_weights))
            h = h * it / (it + 1)
            h += max_value * _mysign(learners[it].predict_proba(X)[
                :, -1] * learners[it].classes_[-1] - 1 / 2) / (it + 1)
            adversary.fit(W, T, h)

        self.learners = learners
        return self

    def predict(self, X):
        return np.mean([self.max_abs_value * _mysign(l.predict_proba(X)
                                                     [:, -1] * l.classes_[-1] - 1 / 2) for l in self.learners], axis=0)
