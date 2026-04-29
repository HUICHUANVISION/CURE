#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CURE_full_auto_label_v6_visual.py
(Enhanced version with visualization for all three stages)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, matthews_corrcoef, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.io import arff
import matplotlib.pyplot as plt
import pandas as pd
import csv

# ===============================
# 工具函数
# ===============================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def clean_and_binarize_label(x):
    if isinstance(x, bytes):
        x = x.decode('utf-8', errors='ignore')
    x = str(x).strip().lower()
    if x in ['y','yes','bug','buggy','true','defective','1']:
        return 1
    elif x in ['n','no','nonbug','clean','false','nondefective','0']:
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

def load_multiple_arff_dirs(arff_dir, exclude_file=None):
    all_X, all_y = [], []
    exclude_name = os.path.basename(exclude_file) if exclude_file else None
    for f in sorted(os.listdir(arff_dir)):
        if f.endswith('.arff') and f != exclude_name:
            X, y = load_arff_dataset(os.path.join(arff_dir, f))
            all_X.append(X); all_y.append(y)
    return np.vstack(all_X), np.concatenate(all_y)

def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return {k: 0.0 for k in ["AUC","F1","Recall","MCC","Pd","Pf","GM"]}
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    pd_ = tp / (tp + fn) if (tp + fn) > 0 else 0
    pf_ = fp / (fp + tn) if (fp + tn) > 0 else 0
    gm = np.sqrt(pd_ * (1 - pf_))
    return {"AUC": auc, "F1": f1, "Recall": recall, "MCC": mcc, "Pd": pd_, "Pf": pf_, "GM": gm}

# ===============================
# 模型定义
# ===============================
class LinearAligner(nn.Module):
    def __init__(self, src_dim, tgt_dim):
        super().__init__()
        self.map = nn.Linear(src_dim, tgt_dim)
    def forward(self, x): return self.map(x)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim), nn.ReLU()
        )
    def forward(self, x): return self.net(x)

class Classifier(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(hidden_dim, 1)
        )
    def forward(self, x): return self.net(x)

class Generator(nn.Module):
    def __init__(self, z_dim, label_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + label_dim, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, z, label):
        return self.net(torch.cat([z, label], dim=1))

# ===============================
# MMD 损失
# ===============================
class MMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
    def gaussian_kernel(self, source, target):
        total = torch.cat([source, target], dim=0)
        L2_distance = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(2)
        bandwidth = torch.mean(L2_distance).detach() / (self.kernel_mul ** (self.kernel_num // 2))
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
        return sum(kernel_val)
    def forward(self, source, target):
        batch_size = source.size(0)
        kernels = self.gaussian_kernel(source, target)
        XX, YY = kernels[:batch_size,:batch_size], kernels[batch_size:,batch_size:]
        XY, YX = kernels[:batch_size,batch_size:], kernels[batch_size:,:batch_size]
        return torch.mean(XX + YY - XY - YX)

# ===============================
# Visualization helpers
# ===============================
def visualize_alignment(encoder_src, encoder_tgt, src_X, tgt_X, device, save_path):
    encoder_src.eval(); encoder_tgt.eval()
    with torch.no_grad():
        src_z = encoder_src(torch.tensor(src_X, dtype=torch.float32).to(device)).cpu().numpy()
        tgt_z = encoder_tgt(torch.tensor(tgt_X, dtype=torch.float32).to(device)).cpu().numpy()
    pca = PCA(n_components=2)
    z_all = pca.fit_transform(np.vstack([src_z, tgt_z]))
    src_p, tgt_p = z_all[:len(src_z)], z_all[len(src_z):]
    plt.figure(figsize=(6,5))
    plt.scatter(src_p[:,0], src_p[:,1], alpha=0.5, label='Source Encoded')
    plt.scatter(tgt_p[:,0], tgt_p[:,1], alpha=0.5, label='Target Encoded')
    plt.legend(); plt.title("Stage 1: Latent Alignment (PCA)")
    plt.tight_layout(); plt.savefig(os.path.join(save_path, "stage1_alignment.png")); plt.close()

def visualize_generation(tgt_X, X_syn, save_path):
    pca = PCA(n_components=2)
    combined = np.vstack([tgt_X, X_syn])
    reduced = pca.fit_transform(combined)
    plt.figure(figsize=(6,5))
    plt.scatter(reduced[:len(tgt_X),0], reduced[:len(tgt_X),1], alpha=0.5, label='Real Target')
    plt.scatter(reduced[len(tgt_X):,0], reduced[len(tgt_X):,1], alpha=0.5, label='Synthetic')
    plt.legend(); plt.title("Stage 2: Synthetic vs Real Target (PCA)")
    plt.tight_layout(); plt.savefig(os.path.join(save_path, "stage2_generation.png")); plt.close()

def visualize_classifier(encoder_tgt, clf, tgt_X, tgt_y, save_path, device):
    encoder_tgt.eval(); clf.eval()
    with torch.no_grad():
        z = encoder_tgt(torch.tensor(tgt_X, dtype=torch.float32).to(device)).cpu().numpy()
        logits = clf(torch.tensor(z, dtype=torch.float32).to(device)).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42)
    z_2d = tsne.fit_transform(z)
    plt.figure(figsize=(6,5))
    plt.scatter(z_2d[:,0], z_2d[:,1], c=probs, cmap='coolwarm', s=20)
    plt.colorbar(label="Predicted Defect Probability")
    plt.title("Stage 3: Decision Boundary Visualization (t-SNE)")
    plt.tight_layout(); plt.savefig(os.path.join(save_path, "stage3_classifier.png")); plt.close()

# ===============================
# 主函数
# ===============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arff_src_dirs', required=True)
    parser.add_argument('--src_file', default=None)
    parser.add_argument('--arff_tgt_file', required=True)
    parser.add_argument('--align_epochs', type=int, default=20)
    parser.add_argument('--final_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_gen_per_seed', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--save_dir', default="./runs/demo_v6_vis")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    set_seed(args.seed)

    # --- 数据加载 ---
    print("📂 Loading data...")
    if args.src_file is not None:
        src_X, src_y = load_arff_dataset(os.path.join(args.arff_src_dirs, args.src_file))
    else:
        exclude_name = os.path.basename(args.arff_tgt_file)
        src_X, src_y = load_multiple_arff_dirs(args.arff_src_dirs, exclude_file=args.arff_tgt_file)
    tgt_X, tgt_y = load_arff_dataset(args.arff_tgt_file)

    # --- 维度映射 ---
    aligner = None
    if src_X.shape[1] != tgt_X.shape[1]:
        aligner = LinearAligner(src_X.shape[1], tgt_X.shape[1]).to(device)
        with torch.no_grad(): src_X = aligner(torch.tensor(src_X, dtype=torch.float32)).cpu().numpy()

    # --- 模型初始化 ---
    encoder_src = Encoder(tgt_X.shape[1]).to(device)
    encoder_tgt = Encoder(tgt_X.shape[1]).to(device)
    clf = Classifier(64).to(device)
    generator = Generator(16, 1, tgt_X.shape[1]).to(device)

    # === Stage 1: 域对齐 ===
    print("🔁 Stage 1: Aligning domains...")
    mmd_loss = MMDLoss()
    optimizer = torch.optim.Adam(list(encoder_src.parameters()) + list(encoder_tgt.parameters()), lr=1e-3)
    src_sim = torch.randn(tgt_X.shape[0], encoder_src.net[0].in_features, device=device)
    for epoch in range(1, args.align_epochs+1):
        encoder_src.train(); encoder_tgt.train()
        optimizer.zero_grad()
        loss = mmd_loss(encoder_src(src_sim), encoder_tgt(torch.tensor(tgt_X, dtype=torch.float32).to(device)))
        loss.backward(); optimizer.step()
        if epoch % 5 == 0:
            print(f"[Align Epoch {epoch}] MMD Loss={loss.item():.4f}")
    visualize_alignment(encoder_src, encoder_tgt, src_X, tgt_X, device, args.save_dir)

    # === Stage 2: 数据生成 ===
    print("🧬 Stage 2: Generating synthetic data...")
    label_set = torch.tensor([0.0, 1.0], device=device)
    X_syn, y_syn = [], []
    for y in label_set:
        y_tensor = torch.full((args.n_gen_per_seed, 1), y, dtype=torch.float32, device=device)
        z = torch.randn(args.n_gen_per_seed, 16, device=device)
        fake = generator(z, y_tensor)
        X_syn.append(fake.detach().cpu().numpy()); y_syn.append(y_tensor.detach().cpu().numpy())
    X_syn, y_syn = np.vstack(X_syn), np.vstack(y_syn)
    visualize_generation(tgt_X, X_syn, args.save_dir)

    # === Stage 3: 分类训练 ===
    print("🏁 Stage 3: Training classifier...")
    all_X = np.concatenate([tgt_X, X_syn])
    all_y = np.concatenate([tgt_y, y_syn.squeeze()])
    loader = DataLoader(TensorDataset(torch.tensor(all_X), torch.tensor(all_y)), batch_size=args.batch_size, shuffle=True)
    opt_clf = torch.optim.Adam(clf.parameters(), lr=5e-4)
    pos_ratio = np.sum(tgt_y == 1) / len(tgt_y)
    pos_weight = torch.tensor([(1 - pos_ratio) / pos_ratio], device=device)
    for epoch in range(1, args.final_epochs + 1):
        clf.train(); total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad(): z = encoder_tgt(x)
            logits = clf(z).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            opt_clf.zero_grad(); loss.backward(); opt_clf.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] Loss={total_loss/len(loader):.4f}")
    visualize_classifier(encoder_tgt, clf, tgt_X, tgt_y, args.save_dir, device)

    print("✅ Visualization complete! Check:", args.save_dir)

if __name__ == "__main__":
    main()