import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import scipy
import scipy.special
from sklearn.linear_model import LassoCV, LogisticRegressionCV, LinearRegression, Lasso, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import torch
import torch.nn as nn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.utils.multiclass import type_of_target

from debiased import DebiasedMoment
from advreisz.linear import SparseLinearAdvRiesz
from advreisz.kernel import AdvNystromKernelReisz, AdvKernelReisz, NystromKernelReisz, KernelReisz
from advreisz.deepreisz import AdvReisz
from advreisz.ensemble import AdvEnsembleReisz, RFrr, interactive_poly_feature_fns
from utilities import AutoKernel, prod_kernel, PluginRR, PluginRR2, FitParamsWrapper



# E[E[Y|D=1, A=1, X] – E[Y|D=0, A=1, X] – (E[Y|D=1, A=0, X] – E[Y|D=0, A=0, X])]
# D is the first column, A is the second, and X is the remaining columns.
def moment_fn(x, test_fn):
    n_obs = x.shape[0]
    if torch.is_tensor(x):
        with torch.no_grad():
            t11 = torch.cat([torch.ones((n_obs, 2)).to(device), x[:, 2:]], dim=1)
            t01 = torch.cat([torch.zeros((n_obs, 1)).to(device), torch.ones((n_obs, 1)).to(device), x[:, 2:]], dim=1)
            t10 = torch.cat([torch.ones((n_obs, 1)).to(device), torch.zeros((n_obs, 1)).to(device), x[:, 2:]], dim=1)
            t00 = torch.cat([torch.zeros((n_obs, 2)).to(device), x[:, 2:]], dim=1)
    else:
        t11 = np.hstack([np.ones((n_obs, 2)), x[:, 2:]])
        t01 = np.hstack([np.zeros((n_obs, 1)), np.ones((n_obs, 1)), x[:, 2:]])
        t10 = np.hstack([np.ones((n_obs, 1)), np.zeros((n_obs, 1)), x[:, 2:]])
        t00 = np.hstack([np.zeros((n_obs, 2)), x[:, 2:]])
    return test_fn(t11) - test_fn(t01) - test_fn(t10) + test_fn(t00)



def get_reg_fn(X, y):
    est = LassoCV(max_iter=10000, random_state=123).fit(X, y)
    return lambda: Lasso(alpha=est.alpha_, max_iter=10000, random_state=123)

def get_splin_fn(X):
    return lambda: SparseLinearAdvRiesz(moment_fn,
                                        featurizer=Pipeline([('p', PolynomialFeatures(degree=2, include_bias=False)),
                                                             ('s', StandardScaler()),
                                                             ('cnt', PolynomialFeatures(degree=1, include_bias=True))]),
                                        n_iter=50000, lambda_theta=0.01, B=10,
                                        tol=0.00001)

def get_advnyskernel_fn(X):
    est = AdvNystromKernelReisz(kernel=lambda X, Y=None: prod_kernel(X, Y=Y, gamma=1.0/X.shape[1]),
                                regm='auto', regl='auto', n_components=100, random_state=123)
    reg = est.opt_reg(X)
    return lambda: AdvNystromKernelReisz(kernel=lambda X, Y=None: prod_kernel(X, Y=Y, gamma=1.0/X.shape[1]),
                                         regm=6*reg, regl=reg, n_components=100, random_state=123)

def get_advkernel_fn(X):
    est = AdvKernelReisz(kernel=lambda X, Y=None: prod_kernel(X, Y=Y, gamma=1.0/X.shape[1]), regm='auto', regl='auto')
    reg = est.opt_reg(X)
    return lambda: AdvKernelReisz(kernel=lambda X, Y=None: prod_kernel(X, Y=Y, gamma=1.0/X.shape[1]), regm=6*reg, regl=reg)

def get_nyskernel_fn(X):
    est = NystromKernelReisz(kernel=lambda X, Y=None: prod_kernel(X, Y=Y, gamma=1.0/X.shape[1]),
                             regl='auto', n_components=100, random_state=123)
    reg = est.opt_reg(X)
    return lambda: NystromKernelReisz(kernel=lambda X, Y=None: prod_kernel(X, Y=Y, gamma=1.0/X.shape[1]),
                                      regl=reg, n_components=100, random_state=123)

def get_kernel_fn(X):
    est = KernelReisz(kernel=lambda X, Y=None: prod_kernel(X, Y=Y, gamma=1.0/X.shape[1]), regl='auto')
    reg = est.opt_reg(X)
    return lambda: KernelReisz(kernel=lambda X, Y=None: prod_kernel(X, Y=Y, gamma=1.0/X.shape[1]), regl=reg)

def get_lg_plugin_fn(X):
    clf = LogisticRegressionCV(cv=3, max_iter=10000, random_state=123)
    C_ = clf.fit(X[:, 1:], X[:, 0]).C_[0]
    return lambda: PluginRR(model_t=LogisticRegression(C=C_, max_iter=10000, random_state=123),
                            min_propensity=0)

def get_rf_plugin_fn(X):
    gcv = GridSearchCV(RandomForestClassifier(bootstrap=True, random_state=123),
                       param_grid={'max_depth': [3, None],
                                   'min_samples_leaf': [10, 50]},
                       scoring='r2',
                       cv=5)
    best_model = clone(gcv.fit(X[:, 1:], X[:, 0]).best_estimator_)
    return lambda: PluginRR(model_t=best_model, min_propensity=0)

def get_lg_plugin_fn2(X):
    clf = LogisticRegressionCV(cv=3, max_iter=10000, random_state=123)
    C_ = clf.fit(X[:, 2:], X[:, 1]).C_[0]
    model_t_A = LogisticRegression(C=C_, max_iter=10000, random_state=123)
    clf = LogisticRegressionCV(cv=3, max_iter=10000, random_state=123)
    C_ = clf.fit(X[:, 1:], X[:, 0]).C_[0]
    model_t_treat = LogisticRegression(C=C_, max_iter=10000, random_state=123)
    return lambda: PluginRR2(model_t_A=model_t_A, model_t_treat=model_t_treat,
                             min_propensity=1e-6)

def get_rf_plugin_fn2(X):
    gcv = GridSearchCV(RandomForestClassifier(bootstrap=True, random_state=123),
                       param_grid={'max_depth': [3, None],
                                   'min_samples_leaf': [10, 50]},
                       scoring='r2',
                       cv=5)
    best_model_A = clone(gcv.fit(X[:, 2:], X[:, 1]).best_estimator_)
    best_model_treat = clone(clone(gcv).fit(X[:, 1:], X[:, 0]).best_estimator_)
    return lambda: PluginRR2(model_t_A=best_model_A,
                             model_t_treat=best_model_treat,
                             min_propensity=1e-6)

device = torch.cuda.current_device() if torch.cuda.is_available() else None

def get_rf_fn(X):
    return lambda: AdvEnsembleReisz(moment_fn=moment_fn, 
                                    n_treatments=2,
                                    max_abs_value=100,
                                    n_iter=200, degree=1)

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

def get_agmm_fn(X):
    torch.manual_seed(123)
    n_hidden = 150
    dropout = 0.5
    return lambda: FitParamsWrapper(AdvReisz(get_learner(X.shape[1], n_hidden, dropout),  # Edited so that #features in is #cols of X
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

print("GPU:", torch.cuda.is_available())



n_splits = 1
res = {}
for q in np.arange(0, 1):  # Data not split up into quantiles
    res[f'q={q}'] = {}
    print(f'Quintile={q}')
    
    # get data
    df = pd.read_stata('https://github.com/gsbDBI/ExperimentData/raw/master/Charitable/RawData/AER%20merged.dta', convert_categoricals=False)
    # df = pd.read_stata('../charitable_giving/Replication/AER merged.dta', convert_categoricals=False)  # For cluster

    df = df.loc[(df['ratio'] == 0) | (df['ratio'] == 1)]
    df = df.drop(['control', 'ratio', 'ratio2', 'ratio3',
                'size', 'size25', 'size50', 'size100', 'sizeno',
                'ask', 'askd1', 'askd2', 'askd3', 'ask1', 'ask2', 'ask3',
                'gave', 'amountchange', 'state50one', 'blue0'], axis=1)
    # state50one just tags one (arbitrary?) observation for each state
    # blue0 and red0 and perfectly collinear (when all variables are nonmissing); bluecty and redcty are not
    df = df.dropna()
    y = df['amount'].values
    X = df[['treatment', 'red0',
            'hpa', 'year5', 'dormant', 'nonlit', 'cases', 'redcty', 'bluecty',
            'pwhite', 'pblack', 'page18_39', 'ave_hh_sz', 'median_hhincome', 'powner', 'psch_atlstba', 'pop_propurban']].values
    
    # scale non-binary variables
    y = y.astype(np.double)
    X = X.astype(np.double)
    idx_nonbi = [i for i in range(2, X.shape[1]) if type_of_target(X[:, i]) != 'binary']  # indices of non-binary variables (first and second columns should be binary)
    X[:, idx_nonbi] = StandardScaler().fit_transform(X[:, idx_nonbi])
    y_scale = np.std(y)
    y = y / y_scale

    # shuffle data
    inds = np.arange(X.shape[0])
    np.random.seed(123)
    np.random.shuffle(inds)
    X, y = X[inds].copy(), y[inds].copy()

    # filter extreme party and treatment propensities
    clf_party = LogisticRegressionCV(cv=5, max_iter=10000, random_state=123).fit(X[:, 2:], X[:, 1])
    clf_treat = LogisticRegressionCV(cv=5, max_iter=10000, random_state=123).fit(X[:, 1:], X[:, 0])
    prop_party = clf_party.predict_proba(X[:, 2:])[:, 1]
    prop_treat = clf_treat.predict_proba(X[:, 1:])[:, 1]
    filt = (prop_party <= .9) & (prop_party >= .1) & (prop_treat <= .9) & (prop_treat >= .1)
    print(X.shape[0], np.sum(filt))
    X, y = X[filt], y[filt]

    for name, get_reisz_fn in [
                               ('splin', get_splin_fn),
                               ('advrkhs', get_advkernel_fn),
                               # ('rkhs', get_kernel_fn),
                               ('nys_advrkhs', get_advnyskernel_fn),
                               # ('nys_rkhs', get_nyskernel_fn),
                               ('plugin_lg', get_lg_plugin_fn2),
                               ('plugin_rf', get_rf_plugin_fn2),
                               ('advnnet', get_agmm_fn),
                               ('advrf', get_rf_fn)
                                ]:
        est = DebiasedMoment(moment_fn=moment_fn,
                             get_reisz_fn=get_reisz_fn,
                             get_reg_fn=get_reg_fn, n_splits=n_splits)
        est.fit(X, y)
        p, s, l, u = est.avg_moment()
        res[f'q={q}'][name] = {'point': p * y_scale, 'stderr': s * y_scale,
                               'lower': l * y_scale, 'upper': u * y_scale}
        p, s, l, u = est.avg_moment(tmle=True)
        res[f'q={q}'][f'{name}_tmle'] = {'point': p * y_scale, 'stderr': s * y_scale,
                                         'lower': l * y_scale, 'upper': u * y_scale}

    print(res[f'q={q}'])
    res[f'q={q}'] = pd.DataFrame(res[f'q={q}']).transpose()
res = pd.concat(res)



# import joblib
# joblib.dump(res, f'advriesz_charitable_ns{n_splits}.joblib')