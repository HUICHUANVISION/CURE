import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import eigh
from .base import HDPModel

class BDA(HDPModel):
    def __init__(self, kernel_type='linear', dim=30, lamb=1.0, mu=0.5, gamma=1.0):
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.mu = mu
        self.gamma = gamma

    def fit(self, Xs, Ys, Xt, Yt=None):
        X = np.vstack((Xs, Xt))
        n, m = len(Xs), len(Xt)
        e = np.vstack((1.0 / n * np.ones((n, 1)), -1.0 / m * np.ones((m, 1))))
        M = e @ e.T * X.shape[1]

        K = rbf_kernel(X, X, gamma=self.gamma) if self.kernel_type == 'rbf' else X @ X.T
        H = np.eye(n + m) - np.ones((n + m, n + m)) / (n + m)
        a = K @ M @ K.T + self.lamb * np.eye(n + m)
        b = K @ H @ K.T
        w, V = eigh(b, a)
        A = V[:, :self.dim]
        self.A = A
        self.K = K
        return self

    def transform(self, X):
        Kx = rbf_kernel(X, self.K, gamma=self.gamma) if self.kernel_type == 'rbf' else X @ self.K.T
        return Kx @ self.A