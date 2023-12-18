# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from scipy.stats.stats import moment
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import BaseEstimator, clone
from econml.grf._base_grf import BaseGRF
from econml.utilities import cross_product
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

class PolyRF(BaseEstimator):
    def __init__(self, *, rf, n_treatments):
        self.rf = rf
        self.n_treatments = n_treatments
        return

    def fit(self, X, y, sample_weight=None):
        self.model_ = Pipeline([
                        ('int', 
                         ColumnTransformer([
                                ('poly', PolynomialFeatures(degree=self.n_treatments,
                                                            interaction_only=True, 
                                                            include_bias=False), 
                                np.arange(self.n_treatments))],
                                remainder='passthrough')),
                          ('rf', clone(self.rf))])
        self.model_.fit(X, y, rf__sample_weight=sample_weight)
        self.classes_ = self.model_.named_steps['rf'].classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)

def _mysign(x):
    return 2 * (x >= 0) - 1


def poly_feature_fns(degree):
    def poly(d, sign=1.0):
        return lambda x: sign * x[:, [0]]**d
    return [poly(t) for t in np.arange(0, degree + 1)]

def interactive_poly_feature_fns(degree, n_treatments):
    def poly(d, ind, sign=1.0):
        return lambda x: sign * x[:, [ind]]**d
    def interactions(d1, d2, ind1, ind2, sign=1.0):
        return lambda x: sign * x[:, [ind1]]**(d1) * x[:, [ind2]]**(d2)
    feat_fns = [poly(0, 0)]
    for i in range(n_treatments):
        feat_fns += [poly(t, i) for t in np.arange(1, degree + 1)]
        for j in np.arange(i + 1, n_treatments):
            feat_fns += [interactions(t, tp, i, j) 
                         for t in np.arange(1, degree + 1)
                         for tp in np.arange(1, degree + 2 - t)]
    return feat_fns


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
        self.n_original_treatments_ = T.shape[1]
        TX = np.hstack([T, X])
        riesz_feats = np.hstack([feat_fn(TX)
                                 for feat_fn in self.riesz_feature_fns])
        mfeats = np.hstack([self.moment_fn(TX, feat_fn)
                            for feat_fn in self.riesz_feature_fns])
        alpha = 2 * (mfeats - y.reshape(-1, 1) * riesz_feats)
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
        point = super().predict(TX_test[:, self.n_original_treatments_:], interval=False)
        return self._translate(point, TX_test)


class AdvEnsembleReisz(BaseEstimator):

    def __init__(self, *, moment_fn, n_treatments=1, 
                 adversary='auto', learner='auto',
                 max_abs_value=4, n_iter=100, degree=2,
                 min_samples_leaf=50,
                 max_depth=5):
        self.moment_fn = moment_fn
        self.n_treatments = n_treatments
        self.adversary = adversary
        self.learner = learner
        self.max_abs_value = max_abs_value
        self.n_iter = n_iter
        self.degree = degree
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth

    def _get_new_adversary(self):
        return RFrr(riesz_feature_fns=interactive_poly_feature_fns(self.degree, self.n_treatments),
                    l2=1e-3,
                    max_depth=max(self.n_treatments, self.max_depth),
                    moment_fn=self.moment_fn, n_estimators=100, max_samples=.5,
                    min_samples_leaf=self.min_samples_leaf, min_impurity_decrease=1e-4, inference=False,
                    honest=True, random_state=123) if self.adversary == 'auto' else clone(self.adversary)

    def _get_new_learner(self):
        return PolyRF(rf=RandomForestClassifier(n_estimators=1,
                                                max_depth=max(self.n_treatments, self.max_depth),
                                                criterion='gini', bootstrap=False,
                                                min_samples_leaf=self.min_samples_leaf,
                                                min_impurity_decrease=1e-4,
                                                random_state=123),
                      n_treatments=self.n_treatments) if self.learner == 'auto' else clone(self.learner)

    def fit(self, X):
        T, W = X[:, :self.n_treatments], X[:, self.n_treatments:]
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


class AdvEnsembleReiszRegVariant(BaseEstimator):

    def __init__(self, *, moment_fn, n_treatments=1, 
                 adversary='auto', learner='auto',
                 max_abs_value=4, n_iter=100, degree=2):
        self.moment_fn = moment_fn
        self.n_treatments = n_treatments
        self.adversary = adversary
        self.learner = learner
        self.max_abs_value = max_abs_value
        self.n_iter = n_iter
        self.degree = degree

    def _get_new_adversary(self):
        return RFrr(riesz_feature_fns=interactive_poly_feature_fns(self.degree, self.n_treatments), l2=1e-3,
                    moment_fn=self.moment_fn, n_estimators=100, max_depth=5, max_samples=.5,
                    min_samples_leaf=20, min_impurity_decrease=1e-4, inference=False,
                    honest=True, random_state=123) if self.adversary == 'auto' else clone(self.adversary)

    def _get_new_learner(self):
        if self.learner == 'auto':
            return Pipeline([
                    ('int', 
                     ColumnTransformer([
                        ('poly', PolynomialFeatures(degree=self.n_treatments,
                                                    interaction_only=True, 
                                                    include_bias=False), 
                         np.arange(self.n_treatments))],
                         remainder='passthrough')),
                    ('rf', RandomForestRegressor(n_estimators=1, max_depth=5,
                        bootstrap=False, min_samples_leaf=50, min_impurity_decrease=1e-4,
                        random_state=123))])
        else:
            clone(self.learner)

    def fit(self, X):
        T, W = X[:, :self.n_treatments], X[:, self.n_treatments:]
        max_value = self.max_abs_value
        adversary = self._get_new_adversary().fit(W, T, np.zeros(T.shape[0]))
        learners = []
        weights = []
        h = 0
        for it in range(self.n_iter):
            test = adversary.predict(X).flatten()
            learners.append(self._get_new_learner().fit(X, test))
            h = h * it / (it + 1)
            preds = learners[it].predict(X)
            weights.append(max_value / np.sqrt(np.mean(preds**2)))
            h += weights[it] * preds / (it + 1)
            adversary.fit(W, T, h)

        self.learners = learners
        self.weights = weights
        return self

    def predict(self, X):
        return np.mean([w * l.predict(X) for (w, l) in zip(self.weights, self.learners)], axis=0)