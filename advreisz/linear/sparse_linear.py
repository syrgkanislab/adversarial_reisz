import numpy as np
from sklearn.linear_model import Lasso, LassoCV, ElasticNet
from sklearn.base import clone
from sklearn.preprocessing import PolynomialFeatures


class _SparseLinearAdvRiesz:

    def __init__(self, moment_fn, featurizer=None,
                 lambda_theta=0.01, B=100, eta_theta='auto', eta_w='auto',
                 n_iter=2000, tol=1e-2, sparsity=None):
        self.moment_fn = moment_fn
        self.featurizer = featurizer if featurizer is not None else PolynomialFeatures(
            degree=1, include_bias=False)
        self.B = B
        self.lambda_theta = lambda_theta
        self.eta_theta = eta_theta
        self.eta_w = eta_w
        self.n_iter = n_iter
        self.tol = tol
        self.sparsity = sparsity

    def predict(self, X):
        return np.dot(self.featurizer.transform(X), self.coef_)

    @property
    def coef(self):
        return self.coef_

    def _check_duality_gap(self, X, moment_vec):
        self.max_response_loss_ = np.linalg.norm(
            moment_vec - np.mean(X * np.dot(X, self.coef_).reshape(-1, 1), axis=0), ord=np.inf)\
            + self.lambda_theta * np.linalg.norm(self.coef_, ord=1)
        self.min_response_loss_ = np.dot(moment_vec, self.w_) + self.B * np.clip(self.lambda_theta
                                                                                 - np.linalg.norm(np.mean(X * np.dot(X, self.w_).reshape(-1, 1),
                                                                                                          axis=0),
                                                                                                  ord=np.inf),
                                                                                 - np.inf, 0)
        self.duality_gap_ = self.max_response_loss_ - self.min_response_loss_
        return self.duality_gap_ < self.tol

    def _post_process(self, X, moment_vec):
        if self.sparsity is not None:
            thresh = 1 / (self.sparsity * (X.shape[0])**(2 / 3))
            filt = (np.abs(self.coef_) < thresh)
            self.coef_[filt] = 0
        self.max_violation_ = np.linalg.norm(
            moment_vec - np.mean(X * np.dot(X, self.coef_).reshape(-1, 1), axis=0), ord=np.inf)
        self._check_duality_gap(X, moment_vec)

    def get_feature_names(self):
        return self.featurizer.get_feature_names()


class SparseLinearAdvRiesz(_SparseLinearAdvRiesz):

    def fit(self, X):

        Xraw = X
        X = self.featurizer.fit_transform(X)
        moment_vec = np.mean(np.array([self.moment_fn(Xraw, lambda x: self.featurizer.transform(x)[:, i])
                                       for i in range(X.shape[1])]), axis=1)

        d_x = X.shape[1]
        B = self.B
        T = self.n_iter
        eta_theta = .5 if self.eta_theta == 'auto' else self.eta_theta
        eta_w = .5 if self.eta_w == 'auto' else self.eta_w
        lambda_theta = self.lambda_theta

        last_gap = np.inf
        t = 1
        while t < T:
            t += 1
            if t == 2:
                self.duality_gaps = []
                theta = np.ones(2 * d_x) * B / (2 * d_x)
                theta_acc = np.ones(2 * d_x) * B / (2 * d_x)
                w = np.ones(2 * d_x) / (2 * d_x)
                w_acc = np.ones(2 * d_x) / (2 * d_x)
                res = np.zeros(2 * d_x)
                res_pre = np.zeros(2 * d_x)
                cors = 0

            # quantities for updating theta
            test_fn = np.dot(X, w[:d_x] - w[d_x:]).reshape(-1, 1)
            cors_t = - np.mean(test_fn * X, axis=0)
            cors += cors_t

            # quantities for updating w
            pred_fn = np.dot(X, theta[:d_x] - theta[d_x:]).reshape(-1, 1)
            res[:d_x] = moment_vec - np.mean(pred_fn * X, axis=0)
            res[d_x:] = - res[:d_x]

            # update theta
            theta[:d_x] = np.exp(-1 - eta_theta *
                                 (cors + cors_t + (t + 1) * lambda_theta))
            theta[d_x:] = np.exp(-1 - eta_theta *
                                 (- cors - cors_t + (t + 1) * lambda_theta))
            normalization = np.sum(theta)
            if normalization > B:
                theta[:] = theta * B / normalization

            # update w
            w[:] = w * \
                np.exp(2 * eta_w * res - eta_w * res_pre)
            w[:] = w / np.sum(w)

            theta_acc = theta_acc * (t - 1) / t + theta / t
            w_acc = w_acc * (t - 1) / t + w / t
            res_pre[:] = res

            if t % 50 == 0:
                self.coef_ = theta_acc[:d_x] - theta_acc[d_x:]
                self.w_ = w_acc[:d_x] - w_acc[d_x:]
                if self._check_duality_gap(X, moment_vec):
                    break
                self.duality_gaps.append(self.duality_gap_)
                if np.isnan(self.duality_gap_):
                    eta_theta /= 2
                    eta_w /= 2
                    t = 1
                elif last_gap < self.duality_gap_:
                    eta_theta /= 1.01
                    eta_w /= 1.01
                last_gap = self.duality_gap_

        self.n_iters_ = t
        self.rho_ = theta_acc
        self.coef_ = theta_acc[:d_x] - theta_acc[d_x:]
        self.w_ = w_acc[:d_x] - w_acc[d_x:]

        self._post_process(X, moment_vec)

        return self
