import numpy as np
import scipy
import scipy.special
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV, Lasso, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity, pairwise_kernels
import joblib
import os
import argparse
from advreisz.kernel import KernelReisz, AdvKernelReisz


def binary_kernel(X, Y=None):
    return 1.0 * (X[:, [0]] == X[:, [0]].T) if Y is None else 1.0 * (X[:, [0]] == Y[:, [0]].T)

def prod_kernel(X, Y=None, *, gamma):

    if hasattr(gamma, '__len__'):
        X = X.copy()
        X[:, 1:] = X[:, 1:] * np.sqrt(gamma).reshape(1, -1)
        if Y is not None:
            Y = Y.copy()
            Y[:, 1:] = Y[:, 1:] * np.sqrt(gamma).reshape(1, -1)
        gamma = 1

    if Y is None:
        return rbf_kernel(X[:, 1:], gamma=gamma) * binary_kernel(X, Y)
    else:
        return rbf_kernel(X[:, 1:], Y[:, 1:], gamma=gamma) * binary_kernel(X, Y)

class AutoKernel:

    def __init__(self, *, type='var'):
        self.type = type
    
    def fit(self, X):
        if self.type == 'var':
            self.gamma_ = 1/ ((X.shape[1] - 1) * np.var(X[:, 1:], axis=0))
        if self.type == 'median':
            self.gamma_ = np.array([1 / ((X.shape[1] - 1) * np.median(np.abs(X[:, [i]] - X[:, [i]].T))**2)
                                    for i in np.arange(1, X.shape[1])])
        self.kernel_ = lambda X, Y=None: prod_kernel(X, Y=Y, gamma=self.gamma_)
        return self

def moment_fn(x, test_fn):
    t1 = test_fn(np.hstack([np.ones((x.shape[0], 1)), x[:, 1:]]))
    t0 = test_fn(np.hstack([np.zeros((x.shape[0], 1)), x[:, 1:]]))
    return t1 - t0

def mean_ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def cfit_exp(it, splin_fn, n_samples=None):
    df = pd.read_csv(f'rahul/sim_{it}.csv', index_col=0)
    y = df['Y'].values
    X = df[['D'] + [f'X{i}' for i in np.arange(1, 11)]].values
    if n_samples is not None:
        X, y = X[:n_samples], y[:n_samples]
    a_pred = np.zeros(X.shape[0])
    reg_pred = np.zeros(X.shape[0])
    moment_pred = np.zeros(X.shape[0])
    for train, test in KFold(n_splits=5).split(X):
        splin = splin_fn().fit(X[train])
        a_pred[test] = splin.predict(X[test])
        est = Pipeline([('p', PolynomialFeatures(degree=1)), ('sc', StandardScaler()),
                        ('l', Lasso(alpha=.1))]).fit(X[train], y[train])
        reg_pred[test] = est.predict(X[test])
        moment_pred[test] = moment_fn(X[test], est.predict)

    return mean_ci(moment_pred + a_pred * (y - reg_pred))

def exp(it, splin_fn, n_samples=None):
    df = pd.read_csv(f'rahul/sim_{it}.csv', index_col=0)
    y = df['Y'].values
    X = df[['D'] + [f'X{i}' for i in np.arange(1, 11)]].values
    if n_samples is not None:
        X, y = X[:n_samples], y[:n_samples]
    splin = splin_fn().fit(X)
    a_test = splin.predict(X)
    est = Pipeline([('p', PolynomialFeatures(degree=1)), ('sc', StandardScaler()),
                    ('l', Lasso(alpha=.1))]).fit(X, y)
    return mean_ci(moment_fn(X, est.predict) + a_test * (y - est.predict(X)))


def all_experiments(n_samples_list, target_dir = '.', kernelid=0):

    if kernelid == 0:
        kernel = lambda X, Y=None: rbf_kernel(X, Y=Y, gamma=.1)
    elif kernelid == 1:
        kernel = lambda X, Y=None: prod_kernel(X, Y=Y, gamma=.1)
    elif kernelid == 2:
        kernel = AutoKernel(type='var')
    elif kernelid == 3:
        kernel = AutoKernel(type='median')
    else:
        raise AttributeError("Not implemented")

    true = 2.2

    for n_samples in n_samples_list:
        reg = (1/n_samples)/100
        splin_fn = lambda: AdvKernelReisz(kernel=kernel, regm=6*reg, regl=reg)
        results = Parallel(n_jobs=-1, verbose=3)(delayed(exp)(it, splin_fn, n_samples) for it in np.arange(1, 101).astype(int))
        joblib.dump(results, os.path.join(target_dir, f'advreisz_nocfit_n_{n_samples}.jbl'))
        results = Parallel(n_jobs=-1, verbose=3)(delayed(cfit_exp)(it, splin_fn, n_samples) for it in np.arange(1, 101).astype(int))
        joblib.dump(results, os.path.join(target_dir, f'advreisz_5fold_cfit_n_{n_samples}.jbl'))

        splin_fn = lambda: KernelReisz(kernel=kernel, regl=6*reg)
        results = Parallel(n_jobs=-1, verbose=3)(delayed(exp)(it, splin_fn, n_samples) for it in np.arange(1, 101).astype(int))
        joblib.dump(results, os.path.join(target_dir, f'kernelreisz_nocfit_n_{n_samples}.jbl'))
        results = Parallel(n_jobs=-1, verbose=3)(delayed(cfit_exp)(it, splin_fn, n_samples) for it in np.arange(1, 101).astype(int))
        joblib.dump(results, os.path.join(target_dir, f'kernelreisz_5fold_cfit_n_{n_samples}.jbl'))

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n_samples", "--n_samples", type=int)
    parser.add_argument("-kernel", "--kernel", type=int, default=0)
    args = parser.parse_args()
    all_experiments([args.n_samples], os.environ['AMLT_OUTPUT_DIR'], args.kernel)