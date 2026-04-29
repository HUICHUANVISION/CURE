import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import eigh
from .base import HDPModel

class TCA(HDPModel):
    def __init__(self, kernel_type='rbf', dim=30, lamb=1.0, gamma=1.0):
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Ys, Xt, Yt=None):
        X = np.vstack((Xs, Xt))
        n, m = len(Xs), len(Xt)
        L = np.zeros((n + m, n + m))
        L[:n, :n] = 1.0 / (n ** 2)
        L[n:, n:] = 1.0 / (m ** 2)
        L[:n, n:] = L[n:, :n] = -1.0 / (n * m)

        K = rbf_kernel(X, X, gamma=self.gamma) if self.kernel_type == 'rbf' else X @ X.T
        H = np.eye(n + m) - np.ones((n + m, n + m)) / (n + m)

        a = K @ L @ K.T + self.lamb * np.eye(n + m)
        b = K @ H @ K.T
        w, V = eigh(b, a)
        A = V[:, :self.dim]
        Z = K @ A
        self.A = A
        self.K = K
        self.X_mean = X.mean(0)
        return self

    def transform(self, X):
        Kx = rbf_kernel(X, self.K, gamma=self.gamma) if self.kernel_type == 'rbf' else X @ self.K.T
        return Kx @ self.A