import numpy as np
from abc import ABC, abstractmethod

class HDPModel(ABC):
    """所有 HDP/迁移模型的基类"""

    @abstractmethod
    def fit(self, Xs, Ys, Xt, Yt=None):
        """训练模型"""
        pass

    @abstractmethod
    def transform(self, X):
        """将数据映射到共享子空间"""
        pass

    def fit_transform(self, Xs, Ys, Xt, Yt=None):
        self.fit(Xs, Ys, Xt, Yt)
        return self.transform(Xs), self.transform(Xt)