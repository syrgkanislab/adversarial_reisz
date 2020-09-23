import numpy as np
from sklearn.linear_model import Lasso, LassoCV, ElasticNet
from sklearn.base import clone
from sklearn.preprocessing import PolynomialFeatures
from .utilities import cross_product
import scipy.special
import warnings


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

    def _covs(self, X, coef, w, cov=None):
        if cov is not None:
            pred_cov = cov @ coef
            test_cov = cov @ w
        else:
            pred_cov = np.mean(
                X * np.dot(X, coef).reshape(-1, 1), axis=0)
            test_cov = np.mean(X * np.dot(X, w).reshape(-1, 1), axis=0)
        return pred_cov, test_cov

    def _check_duality_gap(self, X, moment_vec, cov=None):
        pred_cov, test_cov = self._covs(X, self.coef_, self.w_, cov)
        self.max_response_loss_ = np.linalg.norm(
            moment_vec - pred_cov, ord=np.inf)\
            + self.lambda_theta * np.linalg.norm(self.coef_, ord=1)
        self.min_response_loss_ = np.dot(moment_vec, self.w_) + self.B * np.clip(self.lambda_theta
                                                                                 - np.linalg.norm(test_cov,
                                                                                                  ord=np.inf),
                                                                                 - np.inf, 0)
        self.duality_gap_ = self.max_response_loss_ - self.min_response_loss_
        return self.duality_gap_ < self.tol

    def _post_process(self, X, moment_vec, cov=None):
        if self.sparsity is not None:
            thresh = 1 / (self.sparsity * (X.shape[0])**(2 / 3))
            filt = (np.abs(self.coef_) < thresh)
            self.coef_[filt] = 0
        self.max_violation_ = np.linalg.norm(
            moment_vec - np.mean(X * np.dot(X, self.coef_).reshape(-1, 1), axis=0), ord=np.inf)
        self._check_duality_gap(X, moment_vec, cov)

    def get_feature_names(self):
        return self.featurizer.get_feature_names()


class SparseLinearAdvRiesz(_SparseLinearAdvRiesz):

    def fit(self, X):

        Xraw = X
        X = self.featurizer.fit_transform(X)
        moment_vec = np.mean(np.array([self.moment_fn(Xraw, lambda x: self.featurizer.transform(x)[:, i])
                                       for i in range(X.shape[1])]), axis=1)
        n = X.shape[0]
        d_x = X.shape[1]
        B = self.B
        T = self.n_iter
        lambda_theta = self.lambda_theta
        eta_theta = self.eta_theta
        eta_w = self.eta_w

        if (eta_theta == 'auto') or (eta_w == 'auto') or (d_x < n):
            V = np.mean(cross_product(X, X), axis=0)

        if (eta_theta == 'auto') or (eta_w == 'auto'):
            Vmax = np.linalg.norm(V, ord=np.inf)
            eta_theta = 1 / (4 * Vmax)
            eta_w = 1 / (4 * Vmax)

        self.eta_theta_ = eta_theta
        self.eta_w_ = eta_w

        cov = V.reshape(d_x, d_x) if d_x < n else None

        self.log_theta_list = []
        self.log_w_list = []
        t = 1
        while t < T:
            t += 1
            if t == 2:
                self.duality_gaps = []
                theta = np.ones(2 * d_x) * B / (2 * d_x)
                log_theta = np.zeros(2 * d_x)
                theta_acc = np.ones(2 * d_x) * B / (2 * d_x)
                w = np.ones(2 * d_x) / (2 * d_x)
                log_w = np.zeros(2 * d_x)
                w_acc = np.ones(2 * d_x) / (2 * d_x)
                res = np.zeros(2 * d_x)
                res_pre = np.zeros(2 * d_x)
                cors = 0

            pred_cov, test_cov = self._covs(
                X, theta[:d_x] - theta[d_x:], w[:d_x] - w[d_x:], cov)

            # quantities for updating theta
            cors_t = - test_cov
            cors += cors_t

            # quantities for updating w
            res[:d_x] = moment_vec - pred_cov
            res[d_x:] = - res[:d_x]

            # update theta
            log_theta[:d_x] = -1 - eta_theta * \
                (cors + cors_t + (t + 1) * lambda_theta)
            log_theta[d_x:] = -1 - eta_theta * \
                (- cors - cors_t + (t + 1) * lambda_theta)
            normalization = scipy.special.logsumexp(log_theta)
            if normalization > np.log(B):
                log_theta[:] = log_theta + np.log(B) - normalization
            theta[:] = np.exp(log_theta)

            # update w
            log_w[:] = log_w + 2 * eta_w * res - res_pre
            log_w[:] = log_w - scipy.special.logsumexp(log_w)
            w[:] = np.exp(log_w)

            theta_acc = theta_acc * (t - 1) / t + theta / t
            w_acc = w_acc * (t - 1) / t + w / t
            res_pre[:] = eta_w * res

            if t % 50 == 0:
                self.coef_ = theta_acc[:d_x] - theta_acc[d_x:]
                self.w_ = w_acc[:d_x] - w_acc[d_x:]

                if self._check_duality_gap(X, moment_vec, cov):
                    break
                self.duality_gaps.append(self.duality_gap_)

                if np.isnan(self.duality_gap_):
                    print("found nan ", t)
                    eta_theta /= 2
                    eta_w /= 2
                    t = 1

                self.log_theta_list.append(log_theta.copy())
                self.log_w_list.append(log_w.copy())

        self.n_iters_ = t
        self.rho_ = theta_acc
        self.coef_ = theta_acc[:d_x] - theta_acc[d_x:]
        self.w_ = w_acc[:d_x] - w_acc[d_x:]

        self._post_process(X, moment_vec, cov)

        if self.duality_gap_ > self.tol:
            warnings.warn("Maximum number of iterations reached and duality gap tolerance not achieved. "
                          "Consider increasing the maximum number of iterations.")

        return self
