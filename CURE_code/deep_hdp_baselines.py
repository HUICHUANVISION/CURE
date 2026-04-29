#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deep HDP domain-adaptation baselines for heterogeneous defect prediction.

Implements DANN, ADDA, and Deep CORAL with separate source/target encoders so
source and target projects may have different feature dimensions.
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prep(Xs, Ys, Xt):
    Xs = np.asarray(Xs, dtype=np.float32)
    Xt = np.asarray(Xt, dtype=np.float32)
    Ys = np.asarray(Ys, dtype=np.float32).reshape(-1)
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
    ss, st = StandardScaler(), StandardScaler()
    Xs = ss.fit_transform(Xs).astype(np.float32)
    Xt = st.fit_transform(Xt).astype(np.float32)
    return Xs, Ys, Xt


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return GradientReversalFn.apply(x, self.lambd)


class Encoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 64):
        super().__init__()
        hidden = max(32, min(256, input_dim * 2))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(hidden, output_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class DomainDiscriminator(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, z):
        return self.net(z).view(-1)


class ClassifierHead(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Linear(latent_dim, 1)

    def forward(self, z):
        return self.net(z).view(-1)


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.size(0) < 2 or target.size(0) < 2:
        return torch.tensor(0.0, device=source.device)
    d = source.size(1)
    xm = source - source.mean(dim=0, keepdim=True)
    xmt = target - target.mean(dim=0, keepdim=True)
    cs = xm.t().matmul(xm) / (source.size(0) - 1)
    ct = xmt.t().matmul(xmt) / (target.size(0) - 1)
    return ((cs - ct) ** 2).sum() / (4.0 * d * d)


def _loader(X, y=None, batch_size=64, shuffle=True):
    xt = torch.tensor(X, dtype=torch.float32)
    if y is None:
        ds = TensorDataset(xt)
    else:
        ds = TensorDataset(xt, torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=min(batch_size, max(1, len(X))), shuffle=shuffle, drop_last=False)


def _predict(enc_t, clf, Xt, batch_size=512):
    enc_t.eval(); clf.eval()
    outs = []
    with torch.no_grad():
        for (xb,) in _loader(Xt, None, batch_size, False):
            xb = xb.to(DEVICE)
            outs.append(torch.sigmoid(clf(enc_t(xb))).cpu().numpy())
    return np.concatenate(outs) if outs else np.array([], dtype=np.float32)


def fit_predict_dann(Xs, Ys, Xt, *, seed=0, epochs=50, latent_dim=64, lr=1e-3, lambda_=1.0, lambd=None, **kwargs):
    if "lambda" in kwargs:
        lambda_ = kwargs["lambda"]
    if lambd is not None:
        lambda_ = lambd
    _set_seed(seed)
    Xs, Ys, Xt = _prep(Xs, Ys, Xt)
    if len(np.unique(Ys)) < 2:
        return np.full(len(Xt), float(np.mean(Ys)) if len(Ys) else 0.5)
    es, et = Encoder(Xs.shape[1], latent_dim).to(DEVICE), Encoder(Xt.shape[1], latent_dim).to(DEVICE)
    clf, dom, grl = ClassifierHead(latent_dim).to(DEVICE), DomainDiscriminator(latent_dim).to(DEVICE), GradientReversal(lambda_).to(DEVICE)
    opt = torch.optim.Adam(list(es.parameters()) + list(et.parameters()) + list(clf.parameters()) + list(dom.parameters()), lr=lr, weight_decay=1e-4)
    bs = min(64, max(2, min(len(Xs), len(Xt))))
    sl, tl = _loader(Xs, Ys, bs, True), _loader(Xt, None, bs, True)
    for _ in range(epochs):
        for (xb, yb), (tb,) in zip(sl, iter(tl) if len(tl) >= len(sl) else __import__('itertools').cycle(tl)):
            xb, yb, tb = xb.to(DEVICE), yb.to(DEVICE), tb.to(DEVICE)
            zs, zt = es(xb), et(tb)
            cls_loss = F.binary_cross_entropy_with_logits(clf(zs), yb)
            z = torch.cat([zs, zt], 0)
            dl = torch.cat([torch.zeros(zs.size(0), device=DEVICE), torch.ones(zt.size(0), device=DEVICE)])
            dom_loss = F.binary_cross_entropy_with_logits(dom(grl(z)), dl)
            loss = cls_loss + dom_loss
            opt.zero_grad(); loss.backward(); opt.step()
    return _predict(et, clf, Xt)


def fit_predict_deep_coral(Xs, Ys, Xt, *, seed=0, epochs=50, latent_dim=64, lr=1e-3, lambda_=1.0, **kwargs):
    if "lambda" in kwargs:
        lambda_ = kwargs["lambda"]
    _set_seed(seed)
    Xs, Ys, Xt = _prep(Xs, Ys, Xt)
    if len(np.unique(Ys)) < 2:
        return np.full(len(Xt), float(np.mean(Ys)) if len(Ys) else 0.5)
    es, et = Encoder(Xs.shape[1], latent_dim).to(DEVICE), Encoder(Xt.shape[1], latent_dim).to(DEVICE)
    clf = ClassifierHead(latent_dim).to(DEVICE)
    opt = torch.optim.Adam(list(es.parameters()) + list(et.parameters()) + list(clf.parameters()), lr=lr, weight_decay=1e-4)
    bs = min(64, max(2, min(len(Xs), len(Xt))))
    sl, tl = _loader(Xs, Ys, bs, True), _loader(Xt, None, bs, True)
    import itertools
    for _ in range(epochs):
        for (xb, yb), (tb,) in zip(sl, itertools.cycle(tl)):
            xb, yb, tb = xb.to(DEVICE), yb.to(DEVICE), tb.to(DEVICE)
            zs, zt = es(xb), et(tb)
            loss = F.binary_cross_entropy_with_logits(clf(zs), yb) + lambda_ * coral_loss(zs, zt)
            opt.zero_grad(); loss.backward(); opt.step()
    return _predict(et, clf, Xt)


def fit_predict_adda(Xs, Ys, Xt, *, seed=0, epochs=50, latent_dim=64, lr=1e-3, **kwargs):
    _set_seed(seed)
    Xs, Ys, Xt = _prep(Xs, Ys, Xt)
    if len(np.unique(Ys)) < 2:
        return np.full(len(Xt), float(np.mean(Ys)) if len(Ys) else 0.5)
    es, et = Encoder(Xs.shape[1], latent_dim).to(DEVICE), Encoder(Xt.shape[1], latent_dim).to(DEVICE)
    clf, dom = ClassifierHead(latent_dim).to(DEVICE), DomainDiscriminator(latent_dim).to(DEVICE)
    bs = min(64, max(2, min(len(Xs), len(Xt))))
    sl = _loader(Xs, Ys, bs, True)
    opt_pre = torch.optim.Adam(list(es.parameters()) + list(clf.parameters()), lr=lr, weight_decay=1e-4)
    for _ in range(max(1, epochs // 2)):
        for xb, yb in sl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(clf(es(xb)), yb)
            opt_pre.zero_grad(); loss.backward(); opt_pre.step()
    et.load_state_dict(es.state_dict(), strict=False) if Xs.shape[1] == Xt.shape[1] else None
    opt_d = torch.optim.Adam(dom.parameters(), lr=lr, weight_decay=1e-4)
    opt_t = torch.optim.Adam(et.parameters(), lr=lr, weight_decay=1e-4)
    tl = _loader(Xt, None, bs, True)
    import itertools
    for _ in range(epochs):
        for (xb, _), (tb,) in zip(sl, itertools.cycle(tl)):
            xb, tb = xb.to(DEVICE), tb.to(DEVICE)
            with torch.no_grad(): zs = es(xb)
            zt = et(tb).detach()
            z = torch.cat([zs, zt], 0)
            dl = torch.cat([torch.zeros(zs.size(0), device=DEVICE), torch.ones(zt.size(0), device=DEVICE)])
            dloss = F.binary_cross_entropy_with_logits(dom(z), dl)
            opt_d.zero_grad(); dloss.backward(); opt_d.step()
            zt = et(tb)
            fool = torch.zeros(zt.size(0), device=DEVICE)
            tloss = F.binary_cross_entropy_with_logits(dom(zt), fool)
            opt_t.zero_grad(); tloss.backward(); opt_t.step()
    return _predict(et, clf, Xt)
