import numpy as np
import scipy
import scipy.special
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV, Lasso, LogisticRegressionCV, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity, pairwise_kernels
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
import lightgbm as lgb
import joblib
import os
import argparse
from advreisz.kernel import KernelReisz, AdvKernelReisz, AdvNystromKernelReisz, NystromKernelReisz
from advreisz.linear import SparseLinearAdvRiesz
from advreisz.linear.utilities import SeparateFeaturizer
from debiased import DebiasedMoment
from utilities import PluginRR, prod_kernel, AutoKernel, rmse
from advreisz.ensemble import AdvEnsembleReisz
from econml.sklearn_extensions.model_selection import GridSearchCVList
from pathlib import Path
from utilities import FitParamsWrapper
from advreisz.deepreisz import AdvReisz
import torch
import torch.nn as nn


def get_params(d):
    cov = np.roll(np.eye(d), shift=1, axis=1)
    cov[-1, 0] = 0
    cov = (cov + cov.T)/2 + np.eye(d)
    beta = 1/np.arange(1, d + 1)**2
    return cov, beta

def gen_data(dgp, it, n_samples):
    rs = np.random.RandomState(it)
    if dgp == -1:
        df = pd.read_csv(f'rahul/sim_{it}.csv', index_col=0)
        y = df['Y'].values
        X = df[['D'] + [f'X{i}' for i in np.arange(1, 11)]].values
        if n_samples is not None:
            X, y = X[:n_samples], y[:n_samples]
        cov, beta = get_params(10)
        true_propensity = .05 + .9 * scipy.special.expit(X[:, 1:] @ beta)
        true_reg = 2.2 * X[:, 0] + 1.2 * X[:, 1:] @ beta + X[:, 0] * X[:, 1]
    if dgp == 0:
        d = 10
        cov, beta = get_params(d)
        X = np.zeros((n_samples, d + 1))
        X[:, 1:] = rs.multivariate_normal(np.zeros(d), cov, size=(n_samples,))
        true_propensity = .05 + .9 * scipy.special.expit(X[:, 1:] @ beta)
        X[:, 0] = rs.binomial(1, true_propensity)
        true_reg = 2.2 * X[:, 0] + 1.2 * X[:, 1:] @ beta + X[:, 0] * X[:, 1]
        y = true_reg + rs.normal(0, 1, size=(n_samples,))
    if dgp == 1:
        d = 100
        cov, beta = get_params(d)
        X = np.zeros((n_samples, d + 1))
        X[:, 1:] = rs.multivariate_normal(np.zeros(d), cov, size=(n_samples,))
        true_propensity = .05 + .9 * scipy.special.expit(X[:, 1:] @ beta)
        X[:, 0] = rs.binomial(1, true_propensity)
        true_reg = 2.2 * X[:, 0] + 1.2 * X[:, 1:] @ beta + X[:, 0] * X[:, 1]
        y = true_reg + rs.normal(0, 1, size=(n_samples,))
    if dgp == 2:
        d = 10
        cov, beta = get_params(d)
        X = np.zeros((n_samples, d + 1))
        X[:, 1:] = rs.multivariate_normal(np.zeros(d), cov, size=(n_samples,))
        true_propensity = .05 + .9 * np.clip(X[:, 1:] @ beta, 0, 1)
        X[:, 0] = rs.binomial(1, true_propensity)
        true_reg = 2.2 * X[:, 0] + 1.2 * X[:, 1:] @ beta + X[:, 0] * X[:, 1]
        y = true_reg + rs.normal(0, 1, size=(n_samples,))
    if dgp == 3:
        d = 10
        cov, _ = get_params(d)
        X = np.zeros((n_samples, d + 1))
        X[:, 1:] = rs.multivariate_normal(np.zeros(d), cov, size=(n_samples,))
        true_propensity = .1 + .8 * (X[:, 1] > 0)
        X[:, 0] = rs.binomial(1, true_propensity)
        true_reg = 2.2 * X[:, 0] + 1.2 * (X[:, 1] > 0) + X[:, 0] * X[:, 1]
        y = true_reg + rs.normal(0, 1, size=(n_samples,))
    if dgp == 4:
        d = 10
        cov, beta = get_params(d)
        X = np.zeros((n_samples, d + 1))
        X[:, 1:] = rs.multivariate_normal(np.zeros(d), cov, size=(n_samples,))
        true_propensity = .1 + .8 * (X[:, 1:] @ beta > 0)
        X[:, 0] = rs.binomial(1, true_propensity)
        true_reg = 2.2 * X[:, 0] + 1.2 * (X[:, 1:] @ beta > 0) + X[:, 0] * X[:, 1]
        y = true_reg + rs.normal(0, 1, size=(n_samples,))
    true_reisz = X[:, 0] / true_propensity - (1 - X[:, 0]) / (1 - true_propensity)
    return X, y, true_reg, true_reisz

def moment_fn(x, test_fn):
    if torch.is_tensor(x):
        with torch.no_grad():
            t1 = torch.cat([torch.ones((x.shape[0], 1)), x[:, 1:]], dim=1)
            t0 = torch.cat([torch.zeros((x.shape[0], 1)), x[:, 1:]], dim=1)
    else:
        t1 = np.hstack([np.ones((x.shape[0], 1)), x[:, 1:]])
        t0 = np.hstack([np.zeros((x.shape[0], 1)), x[:, 1:]])
    return test_fn(t1) - test_fn(t0)

def get_kernel(kernelid):
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
    return kernel

def get_ls_reg_fn(X, y):
    est = LassoCV(max_iter=10000, random_state=123).fit(X, y)
    return lambda: Lasso(alpha=est.alpha_, max_iter=10000, random_state=123)

def get_rf_reg_fn(X, y):
    gcv = GridSearchCV(RandomForestRegressor(bootstrap=True, random_state=123),
                       param_grid={'max_depth': [3, None],
                                   'min_samples_leaf': [10, 50]},
                       scoring='r2',
                       cv=5)
    best_model = gcv.fit(X, y).best_estimator_
    return lambda: clone(best_model)

def get_gcv_reg_fn(X, y, *, degrees=[1], verbose=0):
    if len(y.shape) == 2 and y.shape[1] == 1:
        y = y.ravel()
    model = GridSearchCVList([Pipeline([('poly', PolynomialFeatures(include_bias=False,
                                                                   interaction_only=True)),
                                        ('sc', StandardScaler()),
                                        ('ls', Lasso(max_iter=10000, random_state=123))]),
                             RandomForestRegressor(n_estimators=100, min_samples_leaf=20, max_depth=3, random_state=123),
                             lgb.LGBMRegressor(num_leaves=32, random_state=123)],
                             param_grid_list=[{'poly__degree': degrees, 'ls__alpha': np.logspace(-4, 2, 20)},
                                              {'min_weight_fraction_leaf': [.01, .1]},
                                              {'learning_rate': [0.001, 0.1, 0.3], 'max_depth': [3, 5]}],
                             cv=3,
                             scoring='r2',
                             verbose=verbose)
    best_model = model.fit(X, y).best_estimator_
    return lambda: clone(best_model)

def get_advkernel_fn(X):
    est = AdvKernelReisz(kernel=lambda X, Y=None: prod_kernel(X, Y=Y, gamma=1.0/X.shape[1]), regm='auto', regl='auto')
    reg = est.opt_reg(X)
    return lambda: AdvKernelReisz(kernel=lambda X, Y=None: prod_kernel(X, Y=Y, gamma=1.0/X.shape[1]), regm=6*reg, regl=reg)

def get_kernel_fn(X):
    est = KernelReisz(kernel=AutoKernel(type='var'), regl='auto')
    reg = est.opt_reg(X)
    print(est.scores_)
    print(reg)
    return lambda: KernelReisz(kernel=AutoKernel(type='var'), regl=reg)

def get_lg_plugin_fn(X):
    clf = LogisticRegressionCV(cv=3, max_iter=10000, random_state=123)
    C_ = clf.fit(X[:, 1:], X[:, 0]).C_[0]
    model_t_treat = LogisticRegression(C=C_, max_iter=10000, random_state=123)
    return lambda: PluginRR(model_t=model_t_treat,
                            min_propensity=1e-6)

def get_rf_plugin_fn(X):
    gcv = GridSearchCV(RandomForestClassifier(bootstrap=True, random_state=123),
                       param_grid={'max_depth': [3, None],
                                   'min_samples_leaf': [10, 50]},
                       scoring='r2',
                       cv=5)
    best_model_treat = clone(clone(gcv).fit(X[:, 1:], X[:, 0]).best_estimator_)
    return lambda: PluginRR(model_t=best_model_treat,
                             min_propensity=1e-6)

def get_rf_fn(X):
    return lambda: AdvEnsembleReisz(moment_fn=moment_fn,
                                    n_treatments=1,
                                    max_abs_value=15,
                                    n_iter=40,
                                    degree=1)

def get_splin_fn(X):
    return lambda: SparseLinearAdvRiesz(moment_fn,
                                        featurizer=Pipeline([('p', PolynomialFeatures(degree=2, include_bias=False)),
                                                             ('s', StandardScaler()),
                                                             ('cnt', PolynomialFeatures(degree=1, include_bias=True))]),
                                        n_iter=50000, lambda_theta=0.01, B=10,
                                        tol=0.00001)

def get_2dsplin_fn(X):
    feat = Pipeline([('p', PolynomialFeatures(degree=2, include_bias=False))])
    feat.steps.append(('s', StandardScaler()))
    feat.steps.append(('cnt', PolynomialFeatures(degree=1, include_bias=True)))
    return lambda: SparseLinearAdvRiesz(moment_fn, featurizer=feat,
                                        n_iter=50000, lambda_theta=0.01, B=10,
                                        tol=0.00001)

def get_advnyskernel_fn(X):
    n_components = 100
    est = AdvNystromKernelReisz(kernel=lambda X, Y=None: prod_kernel(X, Y=Y, gamma=1.0/X.shape[1]),
                                regm='auto', regl='auto', n_components=n_components, random_state=123)
    reg = est.opt_reg(X)
    return lambda: AdvNystromKernelReisz(kernel=lambda X, Y=None: prod_kernel(X, Y=Y, gamma=1.0/X.shape[1]),
                                         regm=6*reg, regl=reg, n_components=n_components, random_state=123)

def get_advnyskernel_fn_1000(X):
    n_components = 1000
    est = AdvNystromKernelReisz(kernel=lambda X, Y=None: prod_kernel(X, Y=Y, gamma=1.0/X.shape[1]),
                                regm='auto', regl='auto', n_components=n_components, random_state=123)
    reg = est.opt_reg(X)
    return lambda: AdvNystromKernelReisz(kernel=lambda X, Y=None: prod_kernel(X, Y=Y, gamma=1.0/X.shape[1]),
                                         regm=6*reg, regl=reg, n_components=n_components, random_state=123)

def get_nys_kernel_fn(X, kernelid):
    kernel = get_kernel(kernelid)
    est = NystromKernelReisz(kernel=kernel, regl='auto', n_components=50, random_state=123)
    reg = est.opt_reg(X)
    return lambda: NystromKernelReisz(kernel=kernel, regl=reg, n_components=50, random_state=123)

# Returns a deep model for the reisz representer
def get_learner(n_t, n_hidden, p):
    return nn.Sequential(nn.Dropout(p=p), nn.Linear(n_t, n_hidden), nn.LeakyReLU(),
                         nn.Dropout(p=p), nn.Linear(n_hidden, n_hidden), nn.LeakyReLU(),
                         nn.Dropout(p=p), nn.Linear(n_hidden, 1))

# Returns a deep model for the test functions
def get_adversary(n_z, n_hidden, p):
    return nn.Sequential(nn.Dropout(p=p), nn.Linear(n_z, n_hidden), nn.ReLU(),
                         nn.Dropout(p=p), nn.Linear(n_hidden, n_hidden), nn.ReLU(),
                         nn.Dropout(p=p), nn.Linear(n_hidden, 1))

def get_nnet_fn(X):  # "get_agmm_fn"
    n_hidden = 100
    dropout = 0.5
    return lambda: FitParamsWrapper(AdvReisz(get_learner(X.shape[1], n_hidden, dropout),
                                             get_adversary(X.shape[1], n_hidden, dropout),
                                             moment_fn),
                                   val_fr=.2,
                                   preprocess_epochs=200,
                                   earlystop_rounds=100,
                                   store_test_every=20,
                                   learner_lr=1e-4, adversary_lr=1e-4,
                                   learner_l2=6e-4, adversary_l2=1e-4,
                                   n_epochs=1000, bs=100,
                                   logger=None, model_dir=str(Path.home()), device=device, verbose=1)

def debiasedfit(it, dgp, n_samples, get_reisz_fn, get_reg_fn, n_splits):
    X, y, true_reg, true_reisz = gen_data(dgp, it, n_samples)
    est = DebiasedMoment(moment_fn=moment_fn, get_reisz_fn=get_reisz_fn, get_reg_fn=get_reg_fn, n_splits=n_splits)
    est.fit(X, y)
    p, _, l, u = est.avg_moment()
    return p, l, u, rmse(true_reg, est.reg_pred_), rmse(true_reisz, est.reisz_pred_)


################################################
# Posprocess
################################################


def postprocess(prefix_list, n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100):
    true = 2.2
    res = {}
    for n_samples in n_samples_list:
        res[f'n={n_samples}'] = {}
        for name, prefix in prefix_list:
            var = os.path.join(target_dir, f'{prefix}_{dgp}_n_{n_samples}_{start_sample}_{sample_its}.jbl')
            results = joblib.load(var)
            results = np.array(results)
            res[f'n={n_samples}'][name] = {'cov': np.mean((results[:, 1] <= true) & (results[:, 2] >= true)),
                                    'bias': np.mean(results[:, 0] - true),
                                    'rmse': np.sqrt(np.mean((results[:, 0] - true)**2)),
                                    'ci_length': np.mean(results[:, 2] - results[:, 1]),
                                    'reg_rmse': np.mean(results[:, 3]),
                                    'reisz_rmse': np.mean(results[:, 4])}
        res[f'n={n_samples}'] = pd.DataFrame(res[f'n={n_samples}'])
    return pd.concat(res, axis=0)


###################################
# Main Experiment Functions
###################################


def run_experiment(prefix, get_reisz_fn, get_reg_fn, n_splits, n_samples, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs, verbose=3)(delayed(debiasedfit)(it, dgp, n_samples, get_reisz_fn, get_reg_fn, n_splits)
                                                 for it in np.arange(start_sample, start_sample + sample_its).astype(int))
    joblib.dump(results, os.path.join(target_dir, f'{prefix}_{dgp}_n_{n_samples}_{start_sample}_{sample_its}.jbl'))

def advkernel_experiments(n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100, kernelid=0, n_jobs=-1, gcv_reg=False):

    kernel = get_kernel(kernelid)
    
    get_reg_fn = get_ls_reg_fn if not gcv_reg else get_gcv_reg_fn

    for n_samples in n_samples_list:
        reg = (1/n_samples)/100
        get_reisz_fn = lambda X: lambda: AdvKernelReisz(kernel=kernel, regm=6*reg, regl=reg)
        run_experiment('advreisz_nocfit', get_reisz_fn, get_reg_fn, 1, n_samples, dgp=dgp,
                       target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs)
        run_experiment('advreisz_5fold_cfit', get_reisz_fn, get_reg_fn, 5, n_samples, dgp=dgp,
                       target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs)

def advkernel_postprocess(n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100):
    return postprocess([('advrkhs', 'advreisz_nocfit'), ('advrkhs_cfit', 'advreisz_5fold_cfit')],
                       n_samples_list, dgp=dgp, target_dir=target_dir, start_sample=start_sample, sample_its=sample_its)

def kernel_experiments(n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100, kernelid=0, n_jobs=-1, gcv_reg=False):

    kernel = get_kernel(kernelid)
    get_reg_fn = get_ls_reg_fn if not gcv_reg else get_gcv_reg_fn

    for n_samples in n_samples_list:
        reg = (1/n_samples)/100
        get_reisz_fn = lambda X: lambda: KernelReisz(kernel=kernel, regl=np.sqrt(reg))
        run_experiment('kernelreisz_nocfit', get_reisz_fn, get_reg_fn, 1, n_samples, dgp=dgp,
                       target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs)
        run_experiment('kernelreisz_5fold_cfit', get_reisz_fn, get_reg_fn, 5, n_samples, dgp=dgp,
                       target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs)

def kernel_postprocess(n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100):
    return postprocess([('rkhs', 'kernelreisz_nocfit'), ('rkhs_cfit', 'kernelreisz_5fold_cfit')],
                       n_samples_list, dgp=dgp, target_dir=target_dir, start_sample=start_sample, sample_its=sample_its)

def auto_advkernel_experiments(n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100, n_jobs=-1, gcv_reg=False):
    get_reg_fn = get_ls_reg_fn if not gcv_reg else get_gcv_reg_fn
    for n_samples in n_samples_list:
        run_experiment('auto_advreisz_nocfit', get_advkernel_fn, get_reg_fn, 1, n_samples, dgp=dgp,
                       target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs)
        run_experiment('auto_advreisz_5fold_cfit', get_advkernel_fn, get_reg_fn, 5, n_samples, dgp=dgp,
                       target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs)

def auto_advkernel_postprocess(n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100):
    return postprocess([('auto_advrkhs', 'auto_advreisz_nocfit'), ('auto_advrkhs_cfit', 'auto_advreisz_5fold_cfit')],
                       n_samples_list, dgp=dgp, target_dir=target_dir, start_sample=start_sample, sample_its=sample_its)

def auto_kernel_experiments(n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100, n_jobs=-1, gcv_reg=False):
    get_reg_fn = get_ls_reg_fn if not gcv_reg else get_gcv_reg_fn
    for n_samples in n_samples_list:
        run_experiment('auto_kernelreisz_nocfit', get_kernel_fn, get_reg_fn, 1, n_samples, dgp=dgp,
                       target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs)
        run_experiment('auto_kernelreisz_5fold_cfit', get_kernel_fn, get_reg_fn, 5, n_samples, dgp=dgp,
                       target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs)

def auto_kernel_postprocess(n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100):
    return postprocess([('auto_rkhs', 'auto_kernelreisz_nocfit'), ('auto_rkhs_cfit', 'auto_kernelreisz_5fold_cfit')],
                       n_samples_list, dgp=dgp, target_dir=target_dir, start_sample=start_sample, sample_its=sample_its)


def pluginlg_experiments(n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100, n_jobs=-1, gcv_reg=False):
    get_reg_fn = get_ls_reg_fn if not gcv_reg else get_gcv_reg_fn
    for n_samples in n_samples_list:
        run_experiment('plugin_lg_nocfit', get_lg_plugin_fn, get_reg_fn, 1, n_samples, dgp=dgp,
                       target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs)
        run_experiment('plugin_lg_5fold_cfit', get_lg_plugin_fn, get_reg_fn, 5, n_samples, dgp=dgp,
                       target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs)

def pluginlg_postprocess(n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100):
    return postprocess([('pluginlg', 'plugin_lg_nocfit'), ('pluginlg_cfit', 'plugin_lg_5fold_cfit')],
                       n_samples_list, dgp=dgp, target_dir=target_dir, start_sample=start_sample, sample_its=sample_its)


def pluginrf_experiments(n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100, n_jobs=-1, gcv_reg=False):
    get_reg_fn = get_ls_reg_fn if not gcv_reg else get_gcv_reg_fn
    for n_samples in n_samples_list:
        run_experiment('plugin_rf_nocfit', get_rf_plugin_fn, get_reg_fn, 1, n_samples, dgp=dgp,
                       target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs)
        run_experiment('plugin_rf_5fold_cfit', get_rf_plugin_fn, get_reg_fn, 5, n_samples, dgp=dgp,
                       target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs)

def pluginrf_postprocess(n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100):
    return postprocess([('pluginrf', 'plugin_rf_nocfit'), ('pluginrf_cfit', 'plugin_rf_5fold_cfit')],
                       n_samples_list, dgp=dgp, target_dir=target_dir, start_sample=start_sample, sample_its=sample_its)


def nystrom_advkernel_experiments(n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100, kernelid=0, n_jobs=-1, gcv_reg=False):
    get_reg_fn = get_ls_reg_fn if not gcv_reg else get_gcv_reg_fn
    get_reisz_fn = lambda X: get_advnyskernel_fn(X, kernelid)

    for n_samples in n_samples_list:
        run_experiment('advnystromreisz_nocfit', get_reisz_fn, get_reg_fn, 1, n_samples, dgp=dgp,
                       target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs)
        run_experiment('advnystromreisz_5fold_cfit', get_reisz_fn, get_reg_fn, 5, n_samples, dgp=dgp,
                       target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs)

def nystrom_advkernel_postprocess(n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100):
    return postprocess([('nysadvrkhs', 'advnystromreisz_nocfit'), ('nysadvrkhs_cfit', 'advnystromreisz_5fold_cfit')],
                       n_samples_list, dgp=dgp, target_dir=target_dir, start_sample=start_sample, sample_its=sample_its)

def nystrom_advkernel_experiments_1000(n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100, kernelid=0, n_jobs=-1, gcv_reg=False):
    get_reg_fn = get_ls_reg_fn if not gcv_reg else get_gcv_reg_fn
    get_reisz_fn = lambda X: get_advnyskernel_fn_1000(X, kernelid)

    for n_samples in n_samples_list:
        run_experiment('advnystromreisz_nocfit_1000', get_reisz_fn, get_reg_fn, 1, n_samples, dgp=dgp,
                       target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs)
        run_experiment('advnystromreisz_5fold_cfit_1000', get_reisz_fn, get_reg_fn, 5, n_samples, dgp=dgp,
                       target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs)

def nystrom_advkernel_postprocess_1000(n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100):
    return postprocess([('nysadvrkhs_1000', 'advnystromreisz_nocfit_1000'), ('nysadvrkhs_cfit_1000', 'advnystromreisz_5fold_cfit_1000')],
                       n_samples_list, dgp=dgp, target_dir=target_dir, start_sample=start_sample, sample_its=sample_its)

def nystrom_kernel_experiments(n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100, kernelid=0, n_jobs=-1, gcv_reg=False):
    get_reg_fn = get_ls_reg_fn if not gcv_reg else get_gcv_reg_fn
    get_reisz_fn = lambda X: get_nys_kernel_fn(X, kernelid)

    for n_samples in n_samples_list:
        run_experiment('nystormkernelreisz_nocfit', get_reisz_fn, get_reg_fn, 1, n_samples, dgp=dgp,
                       target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs)
        run_experiment('nystormkernelreisz_5fold_cfit', get_reisz_fn, get_reg_fn, 5, n_samples, dgp=dgp,
                       target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs)

def nystrom_kernel_postprocess(n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100):
    return postprocess([('nysrkhs', 'nystormkernelreisz_nocfit'), ('nysrkhs_cfit', 'nystormkernelreisz_5fold_cfit')],
                       n_samples_list, dgp=dgp, target_dir=target_dir, start_sample=start_sample, sample_its=sample_its)


def splin_experiments(n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100, n_jobs=-1, gcv_reg=False):
    get_reg_fn = get_ls_reg_fn if not gcv_reg else get_gcv_reg_fn
    for n_samples in n_samples_list:
        run_experiment('splin_nocfit', get_splin_fn, get_reg_fn, 1, n_samples, dgp=dgp,
                       target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs)
        run_experiment('splin_5fold_cfit', get_splin_fn, get_reg_fn, 5, n_samples, dgp=dgp,
                       target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs)


def splin_postprocess(n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100):
    return postprocess([('splin', 'splin_nocfit'), ('splin_cfit', 'splin_5fold_cfit')],
                       n_samples_list, dgp=dgp, target_dir=target_dir, start_sample=start_sample, sample_its=sample_its)

def poly_splin_experiments(n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100, n_jobs=-1, gcv_reg=False):
    get_reg_fn = get_ls_reg_fn if not gcv_reg else get_gcv_reg_fn
    for n_samples in n_samples_list:
        run_experiment('2dsplin_nocfit', get_2dsplin_fn, get_reg_fn, 1, n_samples, dgp=dgp,
                       target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs)
        run_experiment('2dsplin_5fold_cfit', get_2dsplin_fn, get_reg_fn, 5, n_samples, dgp=dgp,
                       target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs)


def poly_splin_postprocess(n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100):
    return postprocess([('2dsplin', '2dsplin_nocfit'), ('2dsplin_cfit', '2dsplin_5fold_cfit')],
                       n_samples_list, dgp=dgp, target_dir=target_dir, start_sample=start_sample, sample_its=sample_its)


def rf_experiments(n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100, n_jobs=-1, gcv_reg=False):
    get_reg_fn = get_ls_reg_fn if not gcv_reg else get_gcv_reg_fn
    for n_samples in n_samples_list:
        run_experiment('rf_nocfit', get_rf_fn, get_reg_fn, 1, n_samples, dgp=dgp,
                       target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs)
        run_experiment('rf_5fold_cfit', get_rf_fn, get_reg_fn, 5, n_samples, dgp=dgp,
                       target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs)

def rf_postprocess(n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100):
    return postprocess([('rfreisz', 'rf_nocfit'), ('rfreisz_cfit', 'rf_5fold_cfit')],
                       n_samples_list, dgp=dgp, target_dir=target_dir, start_sample=start_sample, sample_its=sample_its)

def nnet_experiments(n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100, n_jobs=-1, gcv_reg=False):
    get_reg_fn = get_ls_reg_fn if not gcv_reg else get_gcv_reg_fn
    for n_samples in n_samples_list:
        run_experiment('nnet_nocfit', get_nnet_fn, get_reg_fn, 1, n_samples, dgp=dgp,
                       target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs)
        run_experiment('nnet_5fold_cfit', get_nnet_fn, get_reg_fn, 5, n_samples, dgp=dgp,
                       target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs)

def nnet_postprocess(n_samples_list, *, dgp=0, target_dir = '.', start_sample=1, sample_its=100):
    return postprocess([('nnet', 'nnet_nocfit'), ('nnet_cfit', 'nnet_5fold_cfit')],
                       n_samples_list, dgp=dgp, target_dir=target_dir, start_sample=start_sample, sample_its=sample_its)


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-dgp", "--dgp", type=int, default=0)
    parser.add_argument("-n_samples", "--n_samples", type=int)
    parser.add_argument("-method", "--method", type=int, default=0)
    parser.add_argument("-kernel", "--kernel", type=int, default=0)
    parser.add_argument("-start_sample", "--start_sample", type=int, default=1)
    parser.add_argument("-sample_its", "--sample_its", type=int, default=100)
    parser.add_argument("-gcv_reg", "--gcv_reg", type=bool, default=True)
    args = parser.parse_args()

    n_samples_list = [args.n_samples]
    dgp = args.dgp
    target_dir = os.environ['AMLT_OUTPUT_DIR']
    kernelid = args.kernel
    start_sample = args.start_sample
    sample_its = args.sample_its
    n_jobs = -1
    gcv = args.gcv_reg

    if args.method == 0:
        advkernel_experiments(n_samples_list, dgp=dgp, target_dir=target_dir, kernelid=kernelid,
                              start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs, gcv_reg=gcv)
    if args.method == 1:
        auto_advkernel_experiments(n_samples_list, dgp=dgp, target_dir=target_dir,
                                   start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs, gcv_reg=gcv)
    if args.method == 2:
        splin_experiments(n_samples_list, dgp=dgp, target_dir=target_dir,
                          start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs, gcv_reg=gcv)
    if args.method == 3:
        pluginlg_experiments(n_samples_list, dgp=dgp, target_dir=target_dir,
                             start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs, gcv_reg=gcv)
    if args.method == 4:
        pluginrf_experiments(n_samples_list, dgp=dgp, target_dir=target_dir,
                             start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs, gcv_reg=gcv)
    if args.method == 5:
        nystrom_advkernel_experiments(n_samples_list, dgp=dgp, target_dir=target_dir, kernelid=kernelid,
                                      start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs, gcv_reg=gcv)
    if args.method == 6:
        rf_experiments(n_samples_list, dgp=dgp, target_dir=target_dir,
                       start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs, gcv_reg=gcv)
    if args.method == 7:
        poly_splin_experiments(n_samples_list, dgp=dgp, target_dir=target_dir,
                               start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs, gcv_reg=gcv)
    if args.method == 8:
        nnet_experiments(n_samples_list, dgp=dgp, target_dir=target_dir,
                         start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs, gcv_reg=gcv)
