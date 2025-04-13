import os
import joblib

import numpy as np
import pandas as pd

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
from sklearn.utils.multiclass import type_of_target

import statsmodels.api as sm  # For probit
from utilities import mean_ci
import scipy.stats as sps
import statistics

from debiased import DebiasedMoment
from advreisz.linear import SparseLinearAdvRiesz
from advreisz.kernel import AdvNystromKernelReisz, AdvKernelReisz, NystromKernelReisz, KernelReisz
from advreisz.deepreisz import AdvReisz
from advreisz.ensemble import AdvEnsembleReisz, AdvEnsembleReiszRegVariant, RFrr, interactive_poly_feature_fns
from utilities import AutoKernel, prod_kernel, PluginRR, PluginRR2, FitParamsWrapper

from experiments import *



def get_reg_fn(X, y):
    est = Pipeline([('p', PolynomialFeatures(degree=2, include_bias=False)),
                    ('s', StandardScaler()),
                    ('lasso', LassoCV(max_iter=10000, random_state=123))])
    est.fit(X, y)

    return lambda: Pipeline([('p', PolynomialFeatures(degree=2, include_bias=False)),
                             ('s', StandardScaler()),
                             ('lasso', Lasso(alpha=est.named_steps['lasso'].alpha_, max_iter=10000, random_state=123))])

def get_splin_fn(X):
    return lambda: SparseLinearAdvRiesz(moment_fn,
                                        featurizer=Pipeline([('p', PolynomialFeatures(degree=2, include_bias=False)),
                                                             ('s', StandardScaler()),
                                                             ('cnt', PolynomialFeatures(degree=1, include_bias=True))]),
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

# def get_advkernel_fn(X):  # Very computationally intensive; more practical to use a Nystrom approximation with 1000 components instead (see above)
#     est = AdvKernelReisz(kernel=lambda X, Y=None: prod_kernel(X, Y=Y, gamma=1.0/X.shape[1]), regm='auto', regl='auto')
#     reg = est.opt_reg(X)
#     return lambda: AdvKernelReisz(kernel=lambda X, Y=None: prod_kernel(X, Y=Y, gamma=1.0/X.shape[1]), regm=6*reg, regl=reg)

device = torch.cuda.current_device() if torch.cuda.is_available() else None
print("GPU:", torch.cuda.is_available())

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

def get_lg_plugin_fn(X):
    clf = LogisticRegressionCV(cv=3, max_iter=10000, random_state=123)
    C_ = clf.fit(X[:, 2:], X[:, 1]).C_[0]
    model_t_A = LogisticRegression(C=C_, max_iter=10000, random_state=123)
    clf = LogisticRegressionCV(cv=3, max_iter=10000, random_state=123)
    C_ = clf.fit(X[:, 1:], X[:, 0]).C_[0]
    model_t_treat = LogisticRegression(C=C_, max_iter=10000, random_state=123)
    return lambda: PluginRR2(model_t_A=model_t_A, model_t_treat=model_t_treat,
                             min_propensity=1e-6)

def get_rf_plugin_fn(X):
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

def get_rf_fn(X):
    return lambda: AdvEnsembleReisz(moment_fn=moment_fn,
                                    n_treatments=2,
                                    max_abs_value=15,  # originally 26
                                    n_iter=40,
                                    degree=1)



# get data
np.random.seed(123)
df = pd.read_stata('AER merged.dta',
                   convert_categoricals=False)
df = df.loc[(df['ratio'] == 0) | (df['ratio'] == 1)]
df = df.drop(['control', 'ratio', 'ratio2', 'ratio3',
            'size', 'size25', 'size50', 'size100', 'sizeno',
            'ask', 'askd1', 'askd2', 'askd3', 'ask1', 'ask2', 'ask3',
            'amountchange', 'state50one', 'blue0'], axis=1)
# state50one just tags one (arbitrary?) observation for each state
# blue0 and red0 and perfectly collinear (when all variables are nonmissing); bluecty and redcty are not
df = df.dropna()
y_amount = df['amount'].values
y_gave = df['gave'].values
X = df[['treatment', 'red0',
        'hpa', 'year5', 'dormant', 'nonlit', 'cases', 'redcty', 'bluecty',
        'pwhite', 'pblack', 'page18_39', 'ave_hh_sz', 'median_hhincome', 'powner', 'psch_atlstba', 'pop_propurban']].values

# synthetic
synthetic = False

if synthetic:

    def true_fn(X):
        return X[:, 0] + X[:, 0] * X[:, 1] + X[:, 1] + .1 * X[:, 2]

    true = np.mean(moment_fn(X, true_fn))
    scale = np.std(true_fn(X)) / 2
    print(true, scale)
    y_amount = true_fn(X) + np.random.normal(0, scale, size=(X.shape[0],))



path = './main_results'
os.chdir(path)
np.random.seed(123)

# scale non-binary variables
y_amount = y_amount.astype(np.double)
y_gave = y_gave.astype(np.double)
X = X.astype(np.double)
idx_nonbi = [i for i in range(2, X.shape[1]) if type_of_target(X[:, i]) != 'binary']  # indices of non-binary variables (first and second columns should be binary)
X[:, idx_nonbi] = StandardScaler().fit_transform(X[:, idx_nonbi])
y_scale = np.std(y_amount)
y_amount = y_amount / y_scale

# shuffle data to get random partitions for cross-validation (doing it once at the start is more efficient than repeatedly doing it; the cross-validation here and elsewhere assumes that the data is shuffled)
inds = np.arange(X.shape[0])
np.random.shuffle(inds)
X, y_amount, y_gave = X[inds].copy(), y_amount[inds].copy(), y_gave[inds].copy()

# drop extreme party and "treatment" (offered-match) propensities
clf_party = LogisticRegressionCV(cv=5, max_iter=10000, random_state=123).fit(X[:, 2:], X[:, 1])
clf_treat = LogisticRegressionCV(cv=5, max_iter=10000, random_state=123).fit(X[:, 1:], X[:, 0])
prop_party = clf_party.predict_proba(X[:, 2:])[:, 1]
prop_treat = clf_treat.predict_proba(X[:, 1:])[:, 1]
filt = (prop_party <= .9) & (prop_party >= .1) & (prop_treat <= .9) & (prop_treat >= .1)
print(X.shape[0], np.sum(filt))
X, y_amount, y_gave = X[filt], y_amount[filt], y_gave[filt]



# This is to replicate set of the regressors using in Table 6, Column 9 in the AER paper
def process_regressors(X_old):
    """Interacts the controls and then adds a constant"""
    return sm.add_constant(np.append(X_old, X_old[:, [0]] * X_old[:, 1:], axis = 1), has_constant = 'add')

def moment_fn_withprocessing(x, test_fn):
    n_obs = x.shape[0]
    t11 = process_regressors(np.hstack([np.ones((n_obs, 2)), x[:, 2:]]))
    t01 = process_regressors(np.hstack([np.zeros((n_obs, 1)), np.ones((n_obs, 1)), x[:, 2:]]))
    t10 = process_regressors(np.hstack([np.ones((n_obs, 1)), np.zeros((n_obs, 1)), x[:, 2:]]))
    t00 = process_regressors(np.hstack([np.zeros((n_obs, 2)), x[:, 2:]]))
    return test_fn(t11) - test_fn(t01) - test_fn(t10) + test_fn(t00)



alpha = 0.05
n_resamples = 1000

# Get data
vars = ['treatment', 'red0', 'pwhite', 'pblack',
        'page18_39', 'ave_hh_sz', 'median_hhincome',
        'powner', 'psch_atlstba', 'pop_propurban']
X_AER = df[vars][filt]
X_AER = X_AER.values

# Calculate critical value
critval = sps.norm(loc=0, scale=1).ppf(1-alpha/2)

# OLS
ols_model = sm.OLS(y_amount*y_scale, process_regressors(X_AER)).fit() # unscale y
p = ols_model.params[11]
s = ols_model.bse[11]
l = p - critval * s
u = p + critval * s
ols_results = {'point': p, 'stderr': s,
               'lower': l, 'upper': u}

# Probit (with bootstrapped standard errors)
def probit_results_fn(y, X):
  probit_model = sm.Probit(y, process_regressors(X)).fit()
  moment_pred = moment_fn_withprocessing(X, probit_model.predict)
  return mean_ci(moment_pred, confidence = 1-alpha)

p, s, l, u = probit_results_fn(y_gave, X_AER)

p_boot_list = []
for i in range(n_resamples):
  idx = np.random.choice(np.arange(len(y_gave)), len(y_gave), replace = True)
  X_boot = X_AER[idx, :]
  y_boot = y_gave[idx]
  p_boot, s_boot, l_boot, u_boot = probit_results_fn(y_boot, X_boot)
  p_boot_list.append(p_boot)

s = statistics.stdev(p_boot_list)
l = p - critval * s
u = p + critval * s

probit_results = {'point': p, 'stderr': s,
                  'lower': l, 'upper': u}

joblib.dump(ols_results, f'ols_results.joblib')
joblib.dump(probit_results, f'probit_results.joblib')



def do_analysis_charitable(y, n_splits, rescale=True):
    res = {}
    for name, get_reisz_fn in [
                                ('plugin_lg', get_lg_plugin_fn),
                                ('plugin_rf', get_rf_plugin_fn),
                                ('splin', get_splin_fn),
                                # ('advrkhs', get_advkernel_fn),
                                ('nys_advrkhs_1000', get_advnyskernel_fn_1000),
                                ('nys_advrkhs', get_advnyskernel_fn),
                                ('advrf', get_rf_fn),
                                ('advnnet', get_agmm_fn),
                                ]:
        est = DebiasedMoment(moment_fn=moment_fn,
                                get_reisz_fn=get_reisz_fn,
                                get_reg_fn=get_reg_fn, n_splits=n_splits)
        est.fit(X, y)
        p, s, l, u = est.avg_moment()
        if rescale==True:
            res[name] = {'point': p * y_scale, 'stderr': s * y_scale,
                                    'lower': l * y_scale, 'upper': u * y_scale}
        else:
            res[name] = {'point': p, 'stderr': s,
                                    'lower': l, 'upper': u}

    res = pd.DataFrame(res).transpose()
    return res

res = do_analysis_charitable(y = y_amount, n_splits = 1)
joblib.dump(res, 'charitable_amount_ns1.joblib')

res = do_analysis_charitable(y = y_amount, n_splits = 5)
joblib.dump(res, 'charitable_amount_ns5.joblib')

res = do_analysis_charitable(y = y_gave, n_splits = 1, rescale=False)
joblib.dump(res, 'charitable_gave_ns1.joblib')

res = do_analysis_charitable(y = y_gave, n_splits = 5, rescale=False)
joblib.dump(res, 'charitable_gave_ns5.joblib')



def do_analysis_charitable(target_dir, dgp, n_samples_list, start_sample, sample_its, n_jobs, gcv):
    pluginlg_experiments(n_samples_list, dgp=dgp, target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs, gcv_reg=gcv)
    pluginrf_experiments(n_samples_list, dgp=dgp, target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs, gcv_reg=gcv)
    splin_experiments(n_samples_list, dgp=dgp, target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs, gcv_reg=gcv)
    advkernel_experiments(n_samples_list, dgp=dgp, target_dir=target_dir, kernelid=2, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs, gcv_reg=gcv)
    nystrom_advkernel_experiments(n_samples_list, dgp=dgp, target_dir=target_dir, kernelid=2, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs, gcv_reg=gcv)
    rf_experiments(n_samples_list, dgp=dgp, target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs, gcv_reg=gcv)
    nnet_experiments(n_samples_list, dgp=dgp, target_dir=target_dir, start_sample=start_sample, sample_its=sample_its, n_jobs=n_jobs, gcv_reg=gcv)

    res = [pluginlg_postprocess(n_samples_list, dgp=dgp, target_dir=target_dir, start_sample=start_sample, sample_its=sample_its),
           pluginrf_postprocess(n_samples_list, dgp=dgp, target_dir=target_dir, start_sample=start_sample, sample_its=sample_its),
           splin_postprocess(n_samples_list, dgp=dgp, target_dir=target_dir, start_sample=start_sample, sample_its=sample_its),
           advkernel_postprocess(n_samples_list, dgp=dgp, target_dir=target_dir, start_sample=start_sample, sample_its=sample_its),
           nystrom_advkernel_postprocess(n_samples_list, dgp=dgp, target_dir=target_dir, start_sample=start_sample, sample_its=sample_its),
           rf_postprocess(n_samples_list, dgp=dgp, target_dir=target_dir, start_sample=start_sample, sample_its=sample_its),
           nnet_postprocess(n_samples_list, dgp=dgp, target_dir=target_dir, start_sample=start_sample, sample_its=sample_its)]
    
    return res

res = do_analysis_charitable(target_dir = '', dgp = 0, n_samples_list = [100], start_sample=0, sample_its = 100, n_jobs = -1, gcv = True)
joblib.dump(res, 'synthetic_dgp0_100.joblib')

res = do_analysis_charitable(target_dir = '', dgp = 0, n_samples_list = [200], start_sample=0, sample_its = 100, n_jobs = -1, gcv = True)
joblib.dump(res, 'synthetic_dgp0_200.joblib')

res = do_analysis_charitable(target_dir = '', dgp = 0, n_samples_list = [500], start_sample=0, sample_its = 100, n_jobs = -1, gcv = True)
joblib.dump(res, 'synthetic_dgp0_500.joblib')

res = do_analysis_charitable(target_dir = '', dgp = 0, n_samples_list = [1000], start_sample=0, sample_its = 100, n_jobs = -1, gcv = True)
joblib.dump(res, 'synthetic_dgp0_1000.joblib')

res = do_analysis_charitable(target_dir = '', dgp = 0, n_samples_list = [2000], start_sample=0, sample_its = 100, n_jobs = -1, gcv = True)
joblib.dump(res, 'synthetic_dgp0_2000.joblib')

res = do_analysis_charitable(target_dir = '', dgp = 3, n_samples_list = [1000], start_sample=0, sample_its = 100, n_jobs = -1, gcv = True)
joblib.dump(res, 'synthetic_dgp3.joblib')

res = do_analysis_charitable(target_dir = '', dgp = 1, n_samples_list = [100], start_sample=0, sample_its = 100, n_jobs = -1, gcv = True)
joblib.dump(res, 'synthetic_dgp1.joblib')