#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rq2_cure_diagnostics.py
RQ2: How does CURE-generated data shape decision boundaries and distributions?

Pipeline:
1) Load target ARFF (and optional source/multi-source if你想用，但此脚本仅需目标域即可)
2) Train a Target Autoencoder -> get latent space Z
3) Class-conditional Gaussian in latent space -> sample synthetic Z_syn, decode -> X_syn
4) BEFORE: train clf on tgt only; AFTER: train clf on tgt+syn
5) Diagnostics:
   - Target-space coverage (kNN-CDF, ε-coverage, KS pass-rate, per-feature Wasserstein, Energy distance)
   - Class separation (Silhouette, CH, DB, Fisher in latent space)
   - Distributional distances (MMD in latent)
   - Decision boundary effects (margins histogram, Brier)
6) Save CSV + plots

Usage example:
python rq2_cure_diagnostics.py \
  --arff_tgt_file path/to/your_target.arff \
  --save_dir runs/rq2_demo \
  --n_syn_per_class 500 \
  --latent_dim 64 --ae_epochs 50 --clf_epochs 30 --batch_size 64 --seed 42
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    brier_score_loss, precision_recall_fscore_support, matthews_corrcoef,
    roc_auc_score, confusion_matrix
)
from scipy.io import arff
from scipy.stats import ks_2samp, wasserstein_distance, energy_distance
import pandas as pd
import matplotlib.pyplot as plt
import csv
np.set_printoptions(suppress=True, linewidth=120)

# --------------------
# Utils
# --------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def clean_and_binarize_label(x):
    if isinstance(x, bytes):
        x = x.decode('utf-8', errors='ignore')
    x = str(x).strip().lower()
    if x in ['y', 'yes', 'bug', 'buggy', 'true', 'defective', '1']:
        return 1
    elif x in ['n', 'no', 'nonbug', 'clean', 'false', 'nondefective', '0']:
        return 0
    try:
        return 1 if float(x) > 0 else 0
    except:
        return 0

def load_arff_dataset(file_path):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    label_col = df.columns[-1]
    y = df[label_col].apply(clean_and_binarize_label).astype(np.float32).values
    X = df.drop(columns=[label_col]).astype(np.float32).values
    print(f"✅ Loaded {os.path.basename(file_path)} | Shape={df.shape}, Label={label_col}, Unique={set(y)}")
    return X, y

def standardize_fit(X):
    mu = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    return mu, std

def standardize_transform(X, mu, std):
    return (X - mu) / std

def inverse_standardize(Xs, mu, std):
    return Xs * std + mu

def evaluate_cls(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    out = {}
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        print("⚠️ Warning: Only one class found in eval.")
        keys = ["AUC","F1","Recall","Precision","MCC","Pd","Pf","GM","Brier"]
        return {k: 0.0 for k in keys}
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    pd_ = tp / (tp + fn) if (tp + fn) > 0 else 0
    pf_ = fp / (fp + tn) if (fp + tn) > 0 else 0
    gm = np.sqrt(max(pd_, 0) * max(1 - pf_, 0))
    out.update(dict(AUC=auc, F1=f1, Recall=recall, Precision=precision, MCC=mcc, Pd=pd_, Pf=pf_, GM=gm, Brier=brier))
    return out

# --------------------
# Models
# --------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden=128, latent=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent)
        )
    def forward(self, x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent, hidden=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )
    def forward(self, z): return self.net(z)

class Classifier(nn.Module):
    def __init__(self, input_dim=64, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(hidden, 1)
        )
    def forward(self, z): return self.net(z)

# --------------------
# Distances & Diagnostics
# --------------------
def knn_min_distances(X_ref, X_query, k=1):
    Xr = torch.tensor(X_ref, dtype=torch.float32)
    Xq = torch.tensor(X_query, dtype=torch.float32)
    d2 = torch.cdist(Xq, Xr, p=2)
    vals, _ = torch.topk(d2, k=k, largest=False)
    return vals[:, -1].cpu().numpy()

def epsilon_coverage(dists, eps):
    return float((dists <= eps).mean())

def ks_pass_rate_per_feature(A, B, alpha=0.05):
    pvals = []
    for j in range(A.shape[1]):
        p = ks_2samp(A[:, j], B[:, j]).pvalue
        pvals.append(p)
    return float((np.array(pvals) > alpha).mean()), pvals

def avg_wasserstein_1d(A, B):
    ws = [wasserstein_distance(A[:, j], B[:, j]) for j in range(A.shape[1])]
    return float(np.mean(ws)), ws

def energy_distance_multivar(A, B):
    # 近似：逐特征 ED 的均值（严格的多维 E-stat 可用第三方包）
    vals = [energy_distance(A[:, j], B[:, j]) for j in range(A.shape[1])]
    return float(np.mean(vals))

@torch.no_grad()
def to_latent(encoder, X, device):
    X = torch.tensor(X, dtype=torch.float32, device=device)
    Z = encoder(X).cpu().numpy()
    return Z

def mmd_rbf(X, Y, kernel_mul=2.0, kernel_num=5):
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    total = torch.cat([X, Y], dim=0)
    L2 = torch.cdist(total, total, p=2) ** 2
    bandwidth = torch.mean(L2).detach() / (kernel_mul ** (kernel_num // 2))
    bws = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    K = sum([torch.exp(-L2 / bw) for bw in bws])
    n, m = X.size(0), Y.size(0)
    Kxx = K[:n, :n].mean()
    Kyy = K[n:, n:].mean()
    Kxy = K[:n, n:].mean()
    return float(Kxx + Kyy - 2 * Kxy)

def fisher_ratio(Z, y):
    y = y.astype(int)
    Z0, Z1 = Z[y==0], Z[y==1]
    mu0, mu1 = Z0.mean(0), Z1.mean(0)
    Sw = np.cov(Z0, rowvar=False) + np.cov(Z1, rowvar=False) + 1e-6*np.eye(Z.shape[1])
    diff = (mu1 - mu0).reshape(-1, 1)
    val = float(diff.T @ np.linalg.inv(Sw) @ diff)
    return val

def class_separation_scores(Z, y):
    y = y.astype(int)
    sil = silhouette_score(Z, y) if len(np.unique(y))==2 else np.nan
    ch = calinski_harabasz_score(Z, y)
    db = davies_bouldin_score(Z, y)
    fr = fisher_ratio(Z, y)
    return {"Silhouette": sil, "CH": ch, "DB": db, "Fisher": fr}

# --------------------
# Plots
# --------------------
def plot_cdf(d1, d2, label1="Before", label2="After", title="NN Distance CDF", save_path=None):
    xs1 = np.sort(d1); xs2 = np.sort(d2)
    cdf1 = np.arange(1, len(xs1)+1)/len(xs1)
    cdf2 = np.arange(1, len(xs2)+1)/len(xs2)
    plt.figure()
    plt.plot(xs1, cdf1, label=label1)
    plt.plot(xs2, cdf2, label=label2)
    plt.xlabel("Distance"); plt.ylabel("CDF"); plt.title(title); plt.legend()
    if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=160)
    plt.close()

def reduce_2d(Z, method="pca"):
    if method=="pca":
        return PCA(n_components=2).fit_transform(Z)
    elif method=="tsne":
        return TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, n_iter=1000).fit_transform(Z)
    else:
        return PCA(n_components=2).fit_transform(Z)

def plot_scatter_2d(Z2, y=None, dom=None, title="", save_path=None):
    plt.figure()
    if y is not None:
        plt.scatter(Z2[:,0], Z2[:,1], s=10, c=y, alpha=0.7)
    elif dom is not None:
        plt.scatter(Z2[:,0], Z2[:,1], s=10, c=dom, alpha=0.7)
    else:
        plt.scatter(Z2[:,0], Z2[:,1], s=10, alpha=0.7)
    plt.title(title)
    if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=160)
    plt.close()

def plot_margins(prob_before, prob_after, save_path=None):
    m_before = np.abs(prob_before - 0.5)
    m_after  = np.abs(prob_after  - 0.5)
    plt.figure()
    plt.hist(m_before, bins=30, alpha=0.6, label="Before")
    plt.hist(m_after,  bins=30, alpha=0.6, label="After")
    plt.title("Margin |p-0.5| Distribution (Target)")
    plt.xlabel("Margin"); plt.ylabel("Count"); plt.legend()
    if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=160)
    plt.close()

# --------------------
# Training
# --------------------
def train_autoencoder(encoder, decoder, loader, device, epochs=50, lr=1e-3):
    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    for ep in range(1, epochs+1):
        encoder.train(); decoder.train()
        tot = 0.0
        for xb, in loader:
            xb = xb.to(device)
            z = encoder(xb)
            xr = decoder(z)
            loss = F.mse_loss(xr, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if ep == 1 or ep % 10 == 0:
            print(f"[AE] Epoch {ep}/{epochs} ReconLoss={tot/len(loader):.5f}")

def train_classifier(encoder, clf, loader, device, epochs=30, lr=5e-4, pos_weight=None):
    encoder.eval(); clf.train()
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    for ep in range(1, epochs+1):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad(): z = encoder(xb)
            logit = clf(z).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if ep == 1 or ep % 10 == 0:
            print(f"[CLF] Epoch {ep}/{epochs} Loss={total/len(loader):.5f}")

@torch.no_grad()
def predict_on_loader(encoder, clf, loader, device):
    encoder.eval(); clf.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        prob = torch.sigmoid(clf(encoder(xb)).squeeze(1)).cpu().numpy()
        ys.append(yb.numpy()); ps.append(prob)
    return np.concatenate(ys), np.concatenate(ps)

# --------------------
# Synthetic data (CURE-style via latent Gaussian)
# --------------------
def class_conditional_latent_sampling(Z, y, n_per_class, shrink=1e-2, seed=42):
    """
    For each class c in {0,1}, estimate mean & covariance (with Tikhonov shrinkage)
    and sample n_per_class points from N(mu_c, Sigma_c + shrink*I).
    """
    rng = np.random.default_rng(seed)
    Zs, Ys = [], []
    for c in [0,1]:
        Zc = Z[y==c]
        if len(Zc) < 2:
            # fallback: small jitter around mean
            mu = Zc.mean(0, keepdims=True) if len(Zc)>0 else rng.normal(0,1,size=(1,Z.shape[1]))
            cov = np.eye(Z.shape[1]) * 0.1
        else:
            mu = Zc.mean(0)
            cov = np.cov(Zc, rowvar=False)
            cov = cov + shrink * np.eye(cov.shape[0])
        sampled = rng.multivariate_normal(mean=mu, cov=cov, size=n_per_class)
        Zs.append(sampled); Ys.append(np.full(n_per_class, c, dtype=np.float32))
    return np.vstack(Zs), np.concatenate(Ys)

# --------------------
# Main
# --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arff_tgt_file', required=True, help='Target ARFF file')
    ap.add_argument('--save_dir', default='runs/rq2_demo')
    ap.add_argument('--latent_dim', type=int, default=64)
    ap.add_argument('--ae_hidden', type=int, default=128)
    ap.add_argument('--ae_epochs', type=int, default=50)
    ap.add_argument('--clf_epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--n_syn_per_class', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--proj_2d', choices=['pca','tsne'], default='pca')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    set_seed(args.seed)

    # 1) Load target data
    X_tgt, y_tgt = load_arff_dataset(args.arff_tgt_file)
    input_dim = X_tgt.shape[1]

    # Standardize features for stable AE training
    mu, std = standardize_fit(X_tgt)
    X_tgt_std = standardize_transform(X_tgt, mu, std).astype(np.float32)

    tgt_ds = TensorDataset(torch.tensor(X_tgt_std))
    tgt_loader_unsup = DataLoader(tgt_ds, batch_size=args.batch_size, shuffle=True)

    # 2) Train Autoencoder on target
    encoder = Encoder(input_dim, hidden=args.ae_hidden, latent=args.latent_dim).to(device)
    decoder = Decoder(latent=args.latent_dim, hidden=args.ae_hidden, output_dim=input_dim).to(device)
    print("🔧 Training target autoencoder...")
    train_autoencoder(encoder, decoder, tgt_loader_unsup, device, epochs=args.ae_epochs, lr=1e-3)

    # Latent of target (for sampling & diagnostics)
    Z_tgt = to_latent(encoder, X_tgt_std, device)

    # 3) Class-conditional latent sampling -> decode -> X_syn
    print("🧬 Generating synthetic samples in latent space...")
    Z_syn, y_syn = class_conditional_latent_sampling(Z_tgt, y_tgt, n_per_class=args.n_syn_per_class, shrink=1e-2, seed=args.seed)
    with torch.no_grad():
        X_syn_std = decoder(torch.tensor(Z_syn, dtype=torch.float32, device=device)).cpu().numpy()
    X_syn = inverse_standardize(X_syn_std, mu, std).astype(np.float32)

    # Before/After train sets (in input feature space)
    X_train_before, y_train_before = X_tgt.astype(np.float32), y_tgt.astype(np.float32)
    X_train_after  = np.concatenate([X_tgt, X_syn]).astype(np.float32)
    y_train_after  = np.concatenate([y_tgt, y_syn]).astype(np.float32)

    # Compute pos_weight on target
    pos_ratio = float((y_tgt == 1).mean())
    pos_weight = torch.tensor([(1 - pos_ratio) / (pos_ratio + 1e-8)], device=device)
    print(f"⚖️ Target positive ratio={pos_ratio:.3f} -> pos_weight={pos_weight.item():.2f}")

    # Build loaders (use standardized inputs for encoder -> consistent with AE)
    def make_loader(X, y=None, shuffle=False):
        Xs = standardize_transform(X, mu, std).astype(np.float32)
        if y is None:
            return DataLoader(TensorDataset(torch.tensor(Xs)), batch_size=args.batch_size, shuffle=shuffle)
        return DataLoader(TensorDataset(torch.tensor(Xs), torch.tensor(y.astype(np.float32))), batch_size=args.batch_size, shuffle=shuffle)

    train_loader_before = make_loader(X_train_before, y_train_before, shuffle=True)
    train_loader_after  = make_loader(X_train_after,  y_train_after,  shuffle=True)
    tgt_loader_eval     = make_loader(X_tgt, y_tgt, shuffle=False)

    # 4) Train classifiers: BEFORE vs AFTER (encoder fixed)
    clf_before = Classifier(input_dim=args.latent_dim).to(device)
    clf_after  = Classifier(input_dim=args.latent_dim).to(device)
    print("🏁 Training BEFORE classifier (target only)...")
    train_classifier(encoder, clf_before, train_loader_before, device, epochs=args.clf_epochs, lr=5e-4, pos_weight=pos_weight)
    print("🏁 Training AFTER  classifier (target + synthetic)...")
    train_classifier(encoder, clf_after,  train_loader_after,  device, epochs=args.clf_epochs, lr=5e-4, pos_weight=pos_weight)

    # Eval on target
    y_true_b, y_prob_b = predict_on_loader(encoder, clf_before, tgt_loader_eval, device)
    y_true_a, y_prob_a = predict_on_loader(encoder, clf_after,  tgt_loader_eval, device)
    metrics_b = evaluate_cls(y_true_b, y_prob_b)
    metrics_a = evaluate_cls(y_true_a, y_prob_a)

    # 5) Diagnostics

    # 5.1 Coverage in input space
    print("📏 Computing coverage metrics...")
    d_before = knn_min_distances(X_train_before, X_tgt)  # NN distance from target to train (before)
    d_after  = knn_min_distances(X_train_after,  X_tgt)  # after augmentation
    plot_cdf(d_before, d_after, "Before", "After", "Target kNN Distance CDF", os.path.join(args.save_dir, "cov_cdf.png"))
    eps = float(np.median(d_before))
    cov_before = epsilon_coverage(d_before, eps)
    cov_after  = epsilon_coverage(d_after,  eps)

    ks_rate_before, _ = ks_pass_rate_per_feature(X_tgt, X_train_before)
    ks_rate_after,  _ = ks_pass_rate_per_feature(X_tgt, X_train_after)
    ws_mean_before, _ = avg_wasserstein_1d(X_tgt, X_train_before)
    ws_mean_after,  _ = avg_wasserstein_1d(X_tgt, X_train_after)
    ed_before = energy_distance_multivar(X_tgt, X_train_before)
    ed_after  = energy_distance_multivar(X_tgt, X_train_after)

    # 5.2 Distribution distances in latent space
    print("📐 Latent-space distances...")
    Z_tr_b = to_latent(encoder, standardize_transform(X_train_before, mu, std), device)
    Z_tr_a = to_latent(encoder, standardize_transform(X_train_after,  mu, std), device)
    mmd_before = mmd_rbf(Z_tgt, Z_tr_b)
    mmd_after  = mmd_rbf(Z_tgt, Z_tr_a)

    # 5.3 Class separation (on target latent)
    print("🔀 Class separation on target latent...")
    sep_scores = class_separation_scores(Z_tgt, y_tgt)

    # 5.4 Decision boundary effect: margins
    print("🗺️ Decision margins...")
    plot_margins(y_prob_b, y_prob_a, save_path=os.path.join(args.save_dir, "margins.png"))

    # 5.5 Visualization: latent scatter
    Z2_tgt = reduce_2d(Z_tgt, method=args.proj_2d)
    plot_scatter_2d(Z2_tgt, y=y_tgt, title=f"Target Latent ({args.proj_2d}) colored by class",
                    save_path=os.path.join(args.save_dir, "latent_tgt_class.png"))

    Z_syn2 = to_latent(encoder, standardize_transform(X_syn, mu, std), device)
    Z2_dom = reduce_2d(np.vstack([Z_tgt, Z_syn2]), method=args.proj_2d)
    dom_labels = np.concatenate([np.zeros(len(Z_tgt)), np.ones(len(Z_syn2))])
    plot_scatter_2d(Z2_dom, dom=dom_labels, title=f"Target vs Synthetic in Latent ({args.proj_2d})",
                    save_path=os.path.join(args.save_dir, "latent_dom.png"))

    # 6) Save CSV (metrics + diagnostics)
    diag_csv = os.path.join(args.save_dir, "rq2_diagnostics.csv")
    with open(diag_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Before", "After"])
        # coverage
        w.writerow(["Coverage@eps(median NN_before)", f"{cov_before:.6f}", f"{cov_after:.6f}"])
        w.writerow(["NN-Dist Median", f"{np.median(d_before):.6f}", f"{np.median(d_after):.6f}"])
        w.writerow(["KS pass rate", f"{ks_rate_before:.6f}", f"{ks_rate_after:.6f}"])
        w.writerow(["Avg Wasserstein(1D)", f"{ws_mean_before:.6f}", f"{ws_mean_after:.6f}"])
        w.writerow(["EnergyDist (feature-wise mean)", f"{ed_before:.6f}", f"{ed_after:.6f}"])
        # latent distances
        w.writerow(["MMD (latent)", f"{mmd_before:.6f}", f"{mmd_after:.6f}"])
        # class separation (latent; single numbers, not before/after)
        w.writerow(["Silhouette (latent target)", "", f"{sep_scores['Silhouette']:.6f}"])
        w.writerow(["Calinski-Harabasz (latent target)", "", f"{sep_scores['CH']:.6f}"])
        w.writerow(["Davies-Bouldin (latent target)", "", f"{sep_scores['DB']:.6f}"])
        w.writerow(["Fisher Ratio (latent target)", "", f"{sep_scores['Fisher']:.6f}"])
        # classifier eval on target
        for k in ["AUC","F1","Recall","Precision","MCC","Pd","Pf","GM","Brier"]:
            w.writerow([f"Target {k}", f"{metrics_b[k]:.6f}", f"{metrics_a[k]:.6f}"])
    print(f"✅ Saved diagnostics to {diag_csv}")

    # Also save a quick summary markdown
    md_path = os.path.join(args.save_dir, "README_RQ2.md")
    with open(md_path, "w") as f:
        f.write("# RQ2 Diagnostics Summary\n\n")
        f.write("## Files\n")
        f.write("- `rq2_diagnostics.csv`\n")
        f.write("- `cov_cdf.png` (Coverage CDF)\n")
        f.write("- `latent_tgt_class.png` (Latent projection by class)\n")
        f.write("- `latent_dom.png` (Target vs Synthetic in latent)\n")
        f.write("- `margins.png` (Decision margins Before/After)\n\n")
        f.write("All distances: feature space standardized by target stats; latent via target autoencoder.\n")
    print(f"📝 Summary written to {md_path}")

if __name__ == "__main__":
    main()