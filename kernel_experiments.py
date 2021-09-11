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
from advreisz.kernel import KernelReisz, AdvKernelReisz, AdvNystromKernelReisz, NystromKernelReisz
from advreisz.linear import SparseLinearAdvRiesz
from debiased import DebiasedMoment


def binary_kernel(X, Y=None):
    return 1.0 * (X[:, [0]] == X[:, [0]].T) if Y is None else 1.0 * (X[:, [0]] == Y[:, [0]].T)

def binary_or_rbf_kernel(X, Y=None, *, gamma):
    return binary_kernel(X, Y=Y) if len(np.unique(X)) == 2 else rbf_kernel(X, Y=Y, gamma=gamma)

def prod_kernel(X, Y=None, *, gamma):

    if hasattr(gamma, '__len__'):
        X = X.copy()
        X[:, 1:] = X[:, 1:] * np.sqrt(gamma).reshape(1, -1)
        if Y is not None:
            Y = Y.copy()
            Y[:, 1:] = Y[:, 1:] * np.sqrt(gamma).reshape(1, -1)
        gamma = 1

    if Y is None:
        # return np.product([binary_or_rbf_kernel(X[:, [t]], gamma=gamma) for t in np.arange(1, X.shape[1])], axis=0)  * binary_kernel(X, Y)
        return rbf_kernel(X[:, 1:], gamma=gamma)  * binary_kernel(X, Y)
    else:
        # return np.product([binary_or_rbf_kernel(X[:, [t]], Y=Y[:, [t]], gamma=gamma) for t in np.arange(1, X.shape[1])], axis=0)  * binary_kernel(X, Y)
        return rbf_kernel(X[:, 1:], Y=Y[:, 1:], gamma=gamma) * binary_kernel(X, Y)


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


def kernel_experiments(n_samples_list, *, target_dir = '.', start_sample=1, sample_its=100, kernelid=0):

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
        results = Parallel(n_jobs=-1, verbose=3)(delayed(exp)(it, splin_fn, n_samples)
                                                 for it in np.arange(start_sample, start_sample + sample_its).astype(int))
        joblib.dump(results, os.path.join(target_dir, f'advreisz_nocfit_n_{n_samples}_{start_sample}_{sample_its}.jbl'))
        results = Parallel(n_jobs=-1, verbose=3)(delayed(cfit_exp)(it, splin_fn, n_samples)
                                                 for it in np.arange(start_sample, start_sample + sample_its).astype(int))
        joblib.dump(results, os.path.join(target_dir, f'advreisz_5fold_cfit_n_{n_samples}_{start_sample}_{sample_its}.jbl'))

        splin_fn = lambda: KernelReisz(kernel=kernel, regl=6*reg)
        results = Parallel(n_jobs=-1, verbose=3)(delayed(exp)(it, splin_fn, n_samples)
                                                 for it in np.arange(start_sample, start_sample + sample_its).astype(int))
        joblib.dump(results, os.path.join(target_dir, f'kernelreisz_nocfit_n_{n_samples}_{start_sample}_{sample_its}.jbl'))
        results = Parallel(n_jobs=-1, verbose=3)(delayed(cfit_exp)(it, splin_fn, n_samples)
                                                 for it in np.arange(start_sample, start_sample + sample_its).astype(int))
        joblib.dump(results, os.path.join(target_dir, f'kernelreisz_5fold_cfit_n_{n_samples}_{start_sample}_{sample_its}.jbl'))


def get_reg_fn(X, y):
    est = LassoCV(max_iter=10000, random_state=123).fit(X, y)
    return lambda: Lasso(alpha=est.alpha_, max_iter=10000, random_state=123)

def get_advkernel_fn(X):
    est = AdvKernelReisz(kernel=AutoKernel(type='var'), regm='auto', regl='auto')
    reg = est.opt_reg(X)
    print(est.scores_)
    print(reg)
    return lambda: AdvKernelReisz(kernel=AutoKernel(type='var'), regm=6*reg, regl=reg)

def get_kernel_fn(X):
    est = KernelReisz(kernel=AutoKernel(type='var'), regl='auto')
    reg = est.opt_reg(X)
    print(est.scores_)
    print(reg)
    return lambda: KernelReisz(kernel=AutoKernel(type='var'), regl=reg)

def debiasedfit(it, n_samples, get_reisz_fn, get_reg_fn, n_splits):
    df = pd.read_csv(f'rahul/sim_{it}.csv', index_col=0)
    y = df['Y'].values
    X = df[['D'] + [f'X{i}' for i in np.arange(1, 11)]].values
    if n_samples is not None:
        X, y = X[:n_samples], y[:n_samples]
    est = DebiasedMoment(moment_fn=moment_fn, get_reisz_fn=get_reisz_fn, get_reg_fn=get_reg_fn, n_splits=n_splits)
    est.fit(X, y)
    p, _, l, u = est.avg_moment()
    return p, l, u

def auto_kernel_experiments(n_samples_list, *, target_dir = '.', start_sample=1, sample_its=100):

    for n_samples in n_samples_list:
        results = Parallel(n_jobs=-1, verbose=3)(delayed(debiasedfit)(it, n_samples, get_advkernel_fn, get_reg_fn, 1)
                                                 for it in np.arange(start_sample, start_sample + sample_its).astype(int))
        joblib.dump(results, os.path.join(target_dir, f'auto_advreisz_nocfit_n_{n_samples}_{start_sample}_{sample_its}.jbl'))
        results = Parallel(n_jobs=-1, verbose=3)(delayed(debiasedfit)(it, n_samples, get_advkernel_fn, get_reg_fn, 5)
                                                 for it in np.arange(start_sample, start_sample + sample_its).astype(int))
        joblib.dump(results, os.path.join(target_dir, f'auto_advreisz_5fold_cfit_n_{n_samples}_{start_sample}_{sample_its}.jbl'))

        results = Parallel(n_jobs=-1, verbose=3)(delayed(debiasedfit)(it, n_samples, get_kernel_fn, get_reg_fn, 1)
                                                 for it in np.arange(start_sample, start_sample + sample_its).astype(int))
        joblib.dump(results, os.path.join(target_dir, f'auto_kernelreisz_nocfit_n_{n_samples}_{start_sample}_{sample_its}.jbl'))
        results = Parallel(n_jobs=-1, verbose=3)(delayed(debiasedfit)(it, n_samples, get_kernel_fn, get_reg_fn, 5)
                                                 for it in np.arange(start_sample, start_sample + sample_its).astype(int))
        joblib.dump(results, os.path.join(target_dir, f'auto_kernelreisz_5fold_cfit_n_{n_samples}_{start_sample}_{sample_its}.jbl'))

def nystrom_kernel_experiments(n_samples_list, *, target_dir = '.', start_sample=1, sample_its=100, kernelid=0):

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
        splin_fn = lambda: AdvNystromKernelReisz(kernel=kernel, regm='auto', regl='auto', n_components=50, random_state=123)
        results = Parallel(n_jobs=-1, verbose=3)(delayed(exp)(it, splin_fn, n_samples)
                                                 for it in np.arange(start_sample, start_sample + sample_its).astype(int))
        joblib.dump(results, os.path.join(target_dir, f'advnystromreisz_nocfit_n_{n_samples}_{start_sample}_{sample_its}.jbl'))
        results = Parallel(n_jobs=-1, verbose=3)(delayed(cfit_exp)(it, splin_fn, n_samples)
                                                 for it in np.arange(start_sample, start_sample + sample_its).astype(int))
        joblib.dump(results, os.path.join(target_dir, f'advnystromreisz_5fold_cfit_n_{n_samples}_{start_sample}_{sample_its}.jbl'))

        splin_fn = lambda: NystromKernelReisz(kernel=kernel, regl='auto', n_components=50, random_state=123)
        results = Parallel(n_jobs=-1, verbose=3)(delayed(exp)(it, splin_fn, n_samples)
                                                 for it in np.arange(start_sample, start_sample + sample_its).astype(int))
        joblib.dump(results, os.path.join(target_dir, f'nystormkernelreisz_nocfit_n_{n_samples}_{start_sample}_{sample_its}.jbl'))
        results = Parallel(n_jobs=-1, verbose=3)(delayed(cfit_exp)(it, splin_fn, n_samples)
                                                 for it in np.arange(start_sample, start_sample + sample_its).astype(int))
        joblib.dump(results, os.path.join(target_dir, f'nystormkernelreisz_5fold_cfit_n_{n_samples}_{start_sample}_{sample_its}.jbl'))

def splin_experiments(n_samples_list, *, target_dir = '.', start_sample=1, sample_its=100):
    feat = Pipeline([('p', PolynomialFeatures(degree=2, include_bias=False))])
    feat.steps.append(('s', StandardScaler()))
    feat.steps.append(('cnt', PolynomialFeatures(degree=1, include_bias=True)))
    splin_fn = lambda: SparseLinearAdvRiesz(moment_fn, featurizer=feat,
                                            n_iter=50000, lambda_theta=0.01, B=10,
                                            tol=0.00001)
    for n_samples in n_samples_list:
        results = Parallel(n_jobs=-1, verbose=3)(delayed(exp)(it, splin_fn, n_samples)
                                                 for it in np.arange(start_sample, start_sample + sample_its).astype(int))
        joblib.dump(results, os.path.join(target_dir, f'splin_nocfit_n_{n_samples}_{start_sample}_{sample_its}.jbl'))
        results = Parallel(n_jobs=-1, verbose=3)(delayed(cfit_exp)(it, splin_fn, n_samples)
                                                 for it in np.arange(start_sample, start_sample + sample_its).astype(int))
        joblib.dump(results, os.path.join(target_dir, f'splin_5fold_cfit_n_{n_samples}_{start_sample}_{sample_its}.jbl'))

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n_samples", "--n_samples", type=int)
    parser.add_argument("-method", "--method", type=int, default=0)
    parser.add_argument("-kernel", "--kernel", type=int, default=0)
    parser.add_argument("-start_sample", "--start_sample", type=int, default=1)
    parser.add_argument("-sample_its", "--sample_its", type=int, default=100)
    args = parser.parse_args()
    if args.method == 0:
        kernel_experiments([args.n_samples],
                            target_dir=os.environ['AMLT_OUTPUT_DIR'],
                            start_sample=args.start_sample,
                            sample_its=args.sample_its,
                            kernelid=args.kernel)
    elif args.method == 1:
        splin_experiments([args.n_samples],
                          target_dir=os.environ['AMLT_OUTPUT_DIR'],
                          start_sample=args.start_sample,
                          sample_its=args.sample_its)
    elif args.method == 2:
        auto_kernel_experiments([args.n_samples],
                                 target_dir=os.environ['AMLT_OUTPUT_DIR'],
                                 start_sample=args.start_sample,
                                 sample_its=args.sample_its)
