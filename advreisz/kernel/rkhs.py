import numpy as np
import scipy.linalg

class AdvKernelReisz:
    
    def __init__(self, *, kernel, regl, regm):
        self.kernel = kernel
        self.regl = regl
        self.regm = regm
    
    def fit(self, X):
        # X is [d; w], i.e. first column is d.
        X1 = np.hstack([np.ones((X.shape[0], 1)), X[:, 1:]])
        X0 = np.hstack([np.zeros((X.shape[0], 1)), X[:, 1:]])
        # Calculate K1, K2, K3, K4
        K1 = self.kernel(X)
        K2 = self.kernel(X, X1) - self.kernel(X, X0)
        K3 = self.kernel(X1, X) - self.kernel(X0, X)
        K4 = self.kernel(X1, X1) - self.kernel(X1, X0) - self.kernel(X0, X1) + self.kernel(X0, X0)
        # Expanded kernel matrix
        K = np.block([[K1, K2], [K3, K4]])

        # Calculate Delta
        n = X.shape[0]
        Delta = np.block([[K1 @ K1, K1 @ K2], [K3 @ K1, K3 @ K2]]) + n * self.regl * K
        invDelta = scipy.linalg.pinv(Delta, atol=1e-6)

        # Calculate Omega
        Sigma = np.block([[K1 @ K1], [K3 @ K1]]).T
        Omega = Sigma.copy()
        Omega -= .5 * Sigma @ invDelta @ np.block([[K1 @ K1, K1 @ K2], [K3 @ K1, K3 @ K2]])
        Omega -= .5 * n * self.regl * Sigma @ invDelta @ K

        # Calculate V
        V = np.zeros(2 * n)
        V[:n] = (1/n) * np.sum(K2, axis=1)
        V[n:] = (1/n) * np.sum(K4, axis=1)

        # Calculate weight for each training sample
        invOmega = scipy.linalg.pinv((1/n) * Omega @ invDelta @ Sigma.T + 2 * self.regm * K1, atol=1e-6)
        self.beta = invOmega @ Omega @ invDelta @ V
        self.Xtrain = X.copy()
        return self
    
    def predict(self, X):
        # calculate test kernel matrix and predictions
        Ktest = self.kernel(X, self.Xtrain)
        return Ktest @ self.beta


# Direct loss
class KernelReisz:

    def __init__(self, *, kernel, regl):
        self.kernel = kernel
        self.regl = regl

    def fit(self, X):
        # X is [d; w], i.e. first column is d.
        X1 = np.hstack([np.ones((X.shape[0], 1)), X[:, 1:]])
        X0 = np.hstack([np.zeros((X.shape[0], 1)), X[:, 1:]])
        # Calculate K1, K2, K3, K4
        K1 = self.kernel(X)
        K2 = self.kernel(X, X1) - self.kernel(X, X0)
        K3 = self.kernel(X1, X) - self.kernel(X0, X)
        K4 = self.kernel(X1, X1) - self.kernel(X1, X0) - self.kernel(X0, X1) + self.kernel(X0, X0)
        # Expanded kernel matrix
        K = np.block([[K1, K2], [K3, K4]])
        
        # Calculate Delta
        n = X.shape[0]
        Delta = np.block([[K1 @ K1, K1 @ K2], [K3 @ K1, K3 @ K2]]) / n + self.regl * K
        invDelta = scipy.linalg.pinv(Delta, atol=1e-6)

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
        Ktest1 = self.kernel(X, self.Xtrain)
        Ktest2 = self.kernel(X, self.X1train) - self.kernel(X, self.X0train)
        return np.block([Ktest1, Ktest2]) @ self.gamma