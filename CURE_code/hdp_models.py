import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError


# ============================================================
# 🔹 通用基类：自动异构特征对齐 + 单类防护
# ============================================================
def align_features(Xs, Xt):
    """通过SVD将源域与目标域特征对齐到相同维度"""
    ds, dt = Xs.shape[1], Xt.shape[1]
    dmin = min(ds, dt)
    Xs_c = Xs - Xs.mean(axis=0, keepdims=True)
    Xt_c = Xt - Xt.mean(axis=0, keepdims=True)

    try:
        _, _, Vx = np.linalg.svd(Xs_c, full_matrices=False)
        _, _, Vy = np.linalg.svd(Xt_c, full_matrices=False)
        Vx = Vx[:dmin, :].T
        Vy = Vy[:dmin, :].T
    except np.linalg.LinAlgError:
        Vx = np.eye(ds, dmin)
        Vy = np.eye(dt, dmin)

    Xs_proj = Xs_c @ Vx
    Xt_proj = Xt_c @ Vy
    return Xs_proj, Xt_proj, Vx, Vy


# ============================================================
# 🔹 模型1：FMT
# ============================================================
class FMTModel:
    """Feature Mapping Transfer"""
    def fit(self, Xs, Ys, Xt):
        Xs_proj, Xt_proj, self.Ws, self.Wt = align_features(Xs, Xt)
        if len(np.unique(Ys)) < 2:
            print("⚠️ Warning: Only one class in Ys, using dummy predictions.")
            self.clf = None
            return self

        self.clf = LogisticRegression(max_iter=200)
        self.clf.fit(Xs_proj, Ys)
        return self

    def predict(self, Xt):
        Xt_c = Xt - Xt.mean(axis=0, keepdims=True)
        Xt_proj = Xt_c @ self.Wt
        if self.clf is None:
            return np.ones(Xt_proj.shape[0]) * 0.5
        try:
            return self.clf.predict_proba(Xt_proj)[:, 1]
        except (NotFittedError, AttributeError):
            return np.ones(Xt_proj.shape[0]) * 0.5


# ============================================================
# 🔹 模型2：CLSUP
# ============================================================
class CLSUPModel:
    """Classifier-Level Supervised Adaptation"""
    def fit(self, Xs, Ys, Xt):
        Xs_proj, Xt_proj, self.Ws, self.Wt = align_features(Xs, Xt)
        if len(np.unique(Ys)) < 2:
            print("⚠️ CLSUP: Only one class in Ys, using dummy predictions.")
            self.clf = None
            return self

        self.clf = LogisticRegression(max_iter=200)
        self.clf.fit(Xs_proj, Ys)
        return self

    def predict(self, Xt):
        Xt_c = Xt - Xt.mean(axis=0, keepdims=True)
        Xt_proj = Xt_c @ self.Wt
        if self.clf is None:
            return np.ones(Xt_proj.shape[0]) * 0.5
        return self.clf.predict_proba(Xt_proj)[:, 1]


# ============================================================
# 🔹 模型3：MSMDA
# ============================================================
class MSMDAModel:
    """Multi-Source MMD-based Adaptation"""
    def fit(self, Xs, Ys, Xt):
        Xs_proj, Xt_proj, self.Ws, self.Wt = align_features(Xs, Xt)

        # MMD-based weighting (简单实现)
        mean_s = np.mean(Xs_proj, axis=0)
        mean_t = np.mean(Xt_proj, axis=0)
        mmd = np.linalg.norm(mean_s - mean_t)
        weight = np.exp(-mmd)

        if len(np.unique(Ys)) < 2:
            print("⚠️ MSMDA: Only one class in Ys, using dummy predictions.")
            self.clf = None
            return self

        self.clf = LogisticRegression(max_iter=200)
        self.clf.fit(Xs_proj * weight, Ys)
        return self

    def predict(self, Xt):
        Xt_c = Xt - Xt.mean(axis=0, keepdims=True)
        Xt_proj = Xt_c @ self.Wt
        if self.clf is None:
            return np.ones(Xt_proj.shape[0]) * 0.5
        return self.clf.predict_proba(Xt_proj)[:, 1]


# ============================================================
# 🔹 模型4：CPDP_IFS
# ============================================================
class CPDP_IFS:
    """Cross-Project Defect Prediction with Instance Filtering + Alignment"""
    def fit(self, Xs, Ys, Xt):
        Xs_proj, Xt_proj, self.Ws, self.Wt = align_features(Xs, Xt)

        mean_t = np.mean(Xt_proj, axis=0)
        sims = np.dot(Xs_proj, mean_t) / (np.linalg.norm(Xs_proj, axis=1) * np.linalg.norm(mean_t) + 1e-8)
        k = int(len(sims) * 0.7)
        keep_idx = np.argsort(sims)[-k:]
        Xs_f, Ys_f = Xs_proj[keep_idx], Ys[keep_idx]

        if len(np.unique(Ys_f)) < 2:
            print("⚠️ CPDP_IFS warning: Only one class after filtering, using dummy predictions.")
            self.clf = None
            return self

        self.clf = LogisticRegression(max_iter=200)
        self.clf.fit(Xs_f, Ys_f)
        return self

    def predict(self, Xt):
        Xt_c = Xt - Xt.mean(axis=0, keepdims=True)
        Xt_proj = Xt_c @ self.Wt
        if self.clf is None:
            return np.ones(Xt_proj.shape[0]) * 0.5
        return self.clf.predict_proba(Xt_proj)[:, 1]


# ============================================================
# 🔹 封装接口：供 run_hdp_experiments.py 调用
# ============================================================
class HDPModel:
    def __init__(self, name):
        self.name = name
        self.model = self._init_model()

    def _init_model(self):
        if self.name == "FMT":
            return FMTModel()
        elif self.name == "CLSUP":
            return CLSUPModel()
        elif self.name == "MSMDA":
            return MSMDAModel()
        elif self.name == "CPDP_IFS":
            return CPDP_IFS()
        else:
            raise ValueError(f"Unknown model: {self.name}")

    def fit_predict(self, Xs, Ys, Xt):
        self.model.fit(Xs, Ys, Xt)
        return self.model.predict(Xt)