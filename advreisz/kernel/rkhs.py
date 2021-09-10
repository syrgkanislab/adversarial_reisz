import numpy as np
from sklearn.base import clone, BaseEstimator
import scipy.linalg
from sklearn.model_selection import train_test_split
from scipy.linalg import pinv
from utilities import Nystroem


def moment_fn(x, test_fn):
    t1 = test_fn(np.hstack([np.ones((x.shape[0], 1)), x[:, 1:]]))
    t0 = test_fn(np.hstack([np.zeros((x.shape[0], 1)), x[:, 1:]]))
    return t1 - t0

class AdvKernelReisz(BaseEstimator):
    
    def __init__(self, *, kernel, regl, regm):
        self.kernel = kernel
        self.regl = regl
        self.regm = regm

    def opt_reg(self, X):
        Xtrain, Xval = train_test_split(X, test_size=.5, random_state=123)
        reglist = np.logspace(-10, 2, 14)
        scores = [AdvKernelReisz(kernel=self.kernel, regm=6*reg, regl=reg).fit(Xtrain).score(Xval)
                  for reg in reglist]
        self.scores_ = scores
        self.reglist_ = reglist
        opt = reglist[np.argmin(scores)]
        return opt

    def fit(self, X):

        if self.regl == 'auto':
            assert self.regm == 'auto', 'if regl==auto, then regm should also be'
            self.regl_ = self.opt_reg(X)
            self.regm_ = 6 * self.regl_
        else:
            self.regl_ = self.regl
            self.regm_ = self.regm

        if hasattr(self.kernel, 'fit'):
            self.kernel_ = self.kernel.fit(X).kernel_
        else:
            self.kernel_ = self.kernel

        # X is [d; w], i.e. first column is d.
        X1 = np.hstack([np.ones((X.shape[0], 1)), X[:, 1:]])
        X0 = np.hstack([np.zeros((X.shape[0], 1)), X[:, 1:]])
        # Calculate K1, K2, K3, K4
        K1 = self.kernel_(X)
        K2 = self.kernel_(X, X1) - self.kernel_(X, X0)
        K3 = self.kernel_(X1, X) - self.kernel_(X0, X)
        K4 = self.kernel_(X1, X1) - self.kernel_(X1, X0) - self.kernel_(X0, X1) + self.kernel_(X0, X0)
        # Expanded kernel matrix
        K = np.block([[K1, K2], [K3, K4]])

        # Calculate Delta
        n = X.shape[0]
        Delta = np.block([[K1 @ K1, K1 @ K2], [K3 @ K1, K3 @ K2]]) + n * self.regl_ * K
        invDelta = pinv(Delta)

        # Calculate Omega
        U = np.block([[K1], [K3]])
        A = U @ K1
        Omega = A.T @ invDelta @ A + 4 * n * self.regm_ * K1

        # Calculate V
        V = np.zeros(2 * n)
        V[:n] = np.sum(K2, axis=1)
        V[n:] = np.sum(K4, axis=1)

        # Calculate weight for each training sample
        invOmega = pinv(Omega)
        self.beta = invOmega @ A.T @ invDelta @ V
        self.gamma = .5 * invDelta @ (V - A @ self.beta)
        self.Xtrain = X.copy()
        self.score_train_ = self.moment_violation(X, self.predict_test)

        return self

    def predict(self, X):
        # calculate test kernel matrix and predictions
        Ktest = self.kernel_(X, self.Xtrain)
        return Ktest @ self.beta
    
    def _predict_test(self, X, Xtrain, gamma):
        X1train = np.hstack([np.ones((Xtrain.shape[0], 1)), Xtrain[:, 1:]])
        X0train = np.hstack([np.zeros((Xtrain.shape[0], 1)), Xtrain[:, 1:]])
        K1test = self.kernel_(X, Xtrain)
        K2test = self.kernel_(X, X1train) - self.kernel_(X, X0train)
        return np.block([K1test, K2test]) @ gamma

    def predict_test(self, X):
        return self._predict_test(X, self.Xtrain, self.gamma)
    
    def moment_violation(self, X, test_fn):
        return np.mean(moment_fn(X, test_fn) - self.predict(X) * test_fn(X))

    def opt_test_fn(self, X, regl):
        X1 = np.hstack([np.ones((X.shape[0], 1)), X[:, 1:]])
        X0 = np.hstack([np.zeros((X.shape[0], 1)), X[:, 1:]])
        # Calculate K1, K2, K3, K4
        K1 = self.kernel_(X)
        K2 = self.kernel_(X, X1) - self.kernel_(X, X0)
        K3 = self.kernel_(X1, X) - self.kernel_(X0, X)
        K4 = self.kernel_(X1, X1) - self.kernel_(X1, X0) - self.kernel_(X0, X1) + self.kernel_(X0, X0)
        K = np.block([[K1, K2], [K3, K4]])
        n = X.shape[0]
        Delta = np.block([[K1 @ K1, K1 @ K2], [K3 @ K1, K3 @ K2]]) + n * regl * K
        invDelta = pinv(Delta)
        V = np.zeros(2 * n)
        V[:n] = np.sum(K2, axis=1)
        V[n:] = np.sum(K4, axis=1)
        gamma = .5 * invDelta @ (V - np.block([[K1], [K3]]) @ self.predict(X))
        return lambda x: self._predict_test(x, X, gamma)

    def max_moment_violation(self, X, regl):
        return self.moment_violation(X, self.opt_test_fn(X, regl))

    def score(self, X):
        Xval1, Xval2 = train_test_split(X, test_size=.5, random_state=123)
        reglist = np.logspace(-8, 2, 12)
        scores = [(self.moment_violation(Xval2, self.opt_test_fn(Xval1, regl)) + 
                   self.moment_violation(Xval1, self.opt_test_fn(Xval2, regl)))/2
                  for regl in reglist]
        return np.max(scores)


# Direct loss
class KernelReisz(BaseEstimator):

    def __init__(self, *, kernel, regl):
        self.kernel = kernel
        self.regl = regl
    
    def opt_reg(self, X):
        Xtrain, Xval = train_test_split(X, test_size=.5, random_state=123)
        reglist = np.logspace(-8, 2, 12)
        scores = [KernelReisz(kernel=self.kernel, regl=reg).fit(Xtrain).score(Xval)
                  for reg in reglist]
        self.scores_ = scores
        self.reglist_ = reglist
        opt = reglist[np.argmin(scores)]
        return opt

    def fit(self, X):

        if self.regl == 'auto':
            self.regl_ = self.opt_reg(X)
        else:
            self.regl_ = self.regl

        if hasattr(self.kernel, 'fit'):
            self.kernel_ = self.kernel.fit(X).kernel_
        else:
            self.kernel_ = self.kernel

        # X is [d; w], i.e. first column is d.
        X1 = np.hstack([np.ones((X.shape[0], 1)), X[:, 1:]])
        X0 = np.hstack([np.zeros((X.shape[0], 1)), X[:, 1:]])
        # Calculate K1, K2, K3, K4
        K1 = self.kernel_(X)
        K2 = self.kernel_(X, X1) - self.kernel_(X, X0)
        K3 = self.kernel_(X1, X) - self.kernel_(X0, X)
        K4 = self.kernel_(X1, X1) - self.kernel_(X1, X0) - self.kernel_(X0, X1) + self.kernel_(X0, X0)
        # Expanded kernel matrix
        K = np.block([[K1, K2], [K3, K4]])
        
        # Calculate Delta
        n = X.shape[0]
        Delta = np.block([[K1 @ K1, K1 @ K2], [K3 @ K1, K3 @ K2]]) / n + self.regl_ * K
        invDelta = pinv(Delta)

        # Calculate V
        V = np.zeros(2 * n)
        V[:n] = (1/n) * np.sum(K2, axis=1)
        V[n:] = (1/n) * np.sum(K4, axis=1)

        # Calculate gamma
        self.gamma = invDelta @ V
        self.Xtrain = X.copy()
        self.X1train = X1.copy()
        self.X0train = X0.copy()
        return self

    def predict(self, X):
        # calculate test kernel matrix and predictions
        Ktest1 = self.kernel_(X, self.Xtrain)
        Ktest2 = self.kernel_(X, self.X1train) - self.kernel_(X, self.X0train)
        return np.block([Ktest1, Ktest2]) @ self.gamma
    
    def score(self, X):
        return np.mean(-2 * moment_fn(X, self.predict) + self.predict(X)**2)



class AdvNystromKernelReisz(BaseEstimator):
    
    def __init__(self, *, kernel, regl, regm, n_components, random_state=None):
        self.kernel = kernel
        self.regl = regl
        self.regm = regm
        self.n_components = n_components
        self.random_state = random_state

    def opt_reg(self, X):
        Xtrain, Xval = train_test_split(X, test_size=.5, random_state=123)
        reglist = np.logspace(-10, 2, 14)
        scores = [AdvNystromKernelReisz(kernel=self.kernel, regm=6*reg,
                                        regl=reg, n_components=self.n_components,
                                        random_state=self.random_state).fit(Xtrain).score(Xval)
                  for reg in reglist]
        self.scores_ = scores
        self.reglist_ = reglist
        opt = reglist[np.argmin(scores)]
        return opt

    def fit(self, X):

        if self.regl == 'auto':
            assert self.regm == 'auto', 'if regl==auto, then regm should also be'
            self.regl_ = self.opt_reg(X)
            self.regm_ = 6 * self.regl_
        else:
            self.regl_ = self.regl
            self.regm_ = self.regm

        if hasattr(self.kernel, 'fit'):
            self.kernel_ = self.kernel.fit(X).kernel_
        else:
            self.kernel_ = self.kernel
        
        self.n_components_ = min(X.shape[0], self.n_components)

        nys = Nystroem(kernel=self.kernel_, n_components=self.n_components_, random_state=self.random_state)
        v = nys.fit_transform(X)

        X1 = np.hstack([np.ones((X.shape[0], 1)), X[:, 1:]])
        X0 = np.hstack([np.zeros((X.shape[0], 1)), X[:, 1:]])
        v1 = nys.transform(X1)
        v0 = nys.transform(X0)
        mu = np.mean(v1 - v0, axis=0)

        n = v.shape[0]
        S = v.T @ v / n
        Sreg = S + self.regl_ * np.eye(S.shape[0])
        invSreg = np.linalg.inv(Sreg)
        Omega = S @ invSreg @ S + 4 * self.regm_ * np.eye(S.shape[0])
        self.beta = np.linalg.inv(Omega) @ S @ invSreg @ mu
        self.gamma = .5 * invSreg @ (mu - S @ self.beta)
        self.nys_ = nys
        self.score_train_ = self.moment_violation(X, self.predict_test)
        return self

    def predict(self, X):
        # calculate test kernel matrix and predictions
        return self.nys_.transform(X) @ self.beta
    
    def _predict_test(self, X, nys, gamma):
        return nys.transform(X) @ gamma

    def predict_test(self, X):
        return self._predict_test(X, self.nys_, self.gamma)
    
    def moment_violation(self, X, test_fn):
        return np.mean(moment_fn(X, test_fn) - self.predict(X) * test_fn(X))

    def opt_test_fn(self, X, regl):
        v = self.nys_.transform(X)

        X1 = np.hstack([np.ones((X.shape[0], 1)), X[:, 1:]])
        X0 = np.hstack([np.zeros((X.shape[0], 1)), X[:, 1:]])
        v1 = self.nys_.transform(X1)
        v0 = self.nys_.transform(X0)
        mu = np.mean(v1 - v0, axis=0)

        n = v.shape[0]
        S = v.T @ v / n
        Sreg = S + regl * np.eye(S.shape[0])
        invSreg = np.linalg.inv(Sreg)
        gamma = .5 * invSreg @ (mu - S @ self.beta)
        return lambda x: self._predict_test(x, self.nys_, gamma)

    def max_moment_violation(self, X, regl):
        return self.moment_violation(X, self.opt_test_fn(X, regl))

    def score(self, X):
        Xval1, Xval2 = train_test_split(X, test_size=.5, random_state=123)
        reglist = np.logspace(-8, 2, 12)
        scores = [(self.moment_violation(Xval2, self.opt_test_fn(Xval1, regl)) + 
                   self.moment_violation(Xval1, self.opt_test_fn(Xval2, regl)))/2
                  for regl in reglist]
        return np.max(scores)



# Direct loss
class NystromKernelReisz(BaseEstimator):

    def __init__(self, *, kernel, regl, n_components, random_state=None):
        self.kernel = kernel
        self.regl = regl
        self.n_components = n_components
        self.random_state = random_state
    
    def opt_reg(self, X):
        Xtrain, Xval = train_test_split(X, test_size=.5, random_state=123)
        reglist = np.logspace(-8, 2, 12)
        scores = [NystromKernelReisz(kernel=self.kernel, regl=reg,
                                     n_components=self.n_components,
                                     random_state=self.random_state).fit(Xtrain).score(Xval)
                  for reg in reglist]
        self.scores_ = scores
        self.reglist_ = reglist
        opt = reglist[np.argmin(scores)]
        return opt

    def fit(self, X):

        if self.regl == 'auto':
            self.regl_ = self.opt_reg(X)
        else:
            self.regl_ = self.regl

        if hasattr(self.kernel, 'fit'):
            self.kernel_ = self.kernel.fit(X).kernel_
        else:
            self.kernel_ = self.kernel

        self.n_components_ = min(X.shape[0], self.n_components)

        nys = Nystroem(kernel=self.kernel_, n_components=self.n_components_, random_state=self.random_state)
        v = nys.fit_transform(X)

        X1 = np.hstack([np.ones((X.shape[0], 1)), X[:, 1:]])
        X0 = np.hstack([np.zeros((X.shape[0], 1)), X[:, 1:]])
        v1 = nys.transform(X1)
        v0 = nys.transform(X0)
        mu = np.mean(v1 - v0, axis=0)
    
        n = v.shape[0]
        S = v.T @ v / n
        Sreg = S + self.regl_ * np.eye(S.shape[0])
        invSreg = np.linalg.inv(Sreg)
        
        # Calculate gamma
        self.gamma = invSreg @ mu
        self.nys_ = nys
        return self

    def predict(self, X):
        # calculate test kernel matrix and predictions
        return self.nys_.transform(X) @ self.gamma
    
    def score(self, X):
        return np.mean(-2 * moment_fn(X, self.predict) + self.predict(X)**2)
