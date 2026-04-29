from sklearn.cross_decomposition import CCA
from .base import HDPModel

class CCAplus(HDPModel):
    def __init__(self, n_components=30):
        self.n_components = n_components

    def fit(self, Xs, Ys, Xt, Yt=None):
        self.model = CCA(n_components=self.n_components)
        self.model.fit(Xs, Xt)
        return self

    def transform(self, X):
        return self.model.transform(X)