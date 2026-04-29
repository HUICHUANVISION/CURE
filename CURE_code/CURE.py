#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CURE_full_auto_label_v6.py (Updated)
Supports both single-source and multi-source CPDP experiments.
Adds --src_file for single-source mode and auto-excludes target file in multi-source mode.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, matthews_corrcoef, confusion_matrix
from scipy.io import arff
import pandas as pd
import csv

# ===============================
# 工具函数
# ===============================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def clean_and_binarize_label(x):
    """统一标签清洗逻辑"""
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

def load_multiple_arff_dirs(arff_dir, exclude_file=None):
    """加载目录下多个 arff 文件，可排除目标文件"""
    all_X, all_y = [], []
    exclude_name = os.path.basename(exclude_file) if exclude_file else None
    for f in sorted(os.listdir(arff_dir)):
        if f.endswith('.arff') and f != exclude_name:
            X, y = load_arff_dataset(os.path.join(arff_dir, f))
            all_X.append(X)
            all_y.append(y)
    return np.vstack(all_X), np.concatenate(all_y)

def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        print("⚠️ Warning: Only one class found.")
        return {k: 0.0 for k in ["AUC","F1","Recall","Precision","MCC","Pd","Pf","GM"]}
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    pd_ = tp / (tp + fn) if (tp + fn) > 0 else 0
    pf_ = fp / (fp + tn) if (fp + tn) > 0 else 0
    gm = np.sqrt(pd_ * (1 - pf_))
    return {"AUC": auc, "F1": f1, "Recall": recall, "Precision": precision,
            "MCC": mcc, "Pd": pd_, "Pf": pf_, "GM": gm}

# ===============================
# 模型模块
# ===============================
class LinearAligner(nn.Module):
    """当源域和目标域维度不匹配时进行线性映射"""
    def __init__(self, src_dim, tgt_dim):
        super().__init__()
        self.map = nn.Linear(src_dim, tgt_dim)
    def forward(self, x):
        return self.map(x)

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
# 训练流程
# ===============================
def train_alignment(encoder_src, encoder_tgt, tgt_X, device, align_epochs=10):
    optimizer = torch.optim.Adam(list(encoder_src.parameters()) + list(encoder_tgt.parameters()), lr=1e-3)
    mmd_loss = MMDLoss()
    src_sim = torch.randn(tgt_X.size(0), encoder_src.net[0].in_features, device=device)
    for epoch in range(1, align_epochs+1):
        encoder_src.train(); encoder_tgt.train()
        optimizer.zero_grad()
        loss = mmd_loss(encoder_src(src_sim), encoder_tgt(tgt_X))
        loss.backward(); optimizer.step()
        print(f"[Align Epoch {epoch}] MMD Loss: {loss.item():.4f}")

def generate_samples(generator, label_set, n_gen, z_dim, device):
    generator.eval(); Xs, Ys = [], []
    for y in label_set:
        y_tensor = torch.full((n_gen, 1), y, dtype=torch.float32, device=device)
        z = torch.randn(n_gen, z_dim, device=device)
        fake = generator(z, y_tensor)
        Xs.append(fake.detach().cpu()); Ys.append(y_tensor.cpu())
    return torch.cat(Xs, 0), torch.cat(Ys, 0)

def train_classifier(encoder, clf, loader, optimizer, device, pos_weight=None):
    encoder.eval(); clf.train(); total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad(): z = encoder(x)
        logits = clf(z).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def predict(encoder, clf, loader, device):
    encoder.eval(); clf.eval(); y_true, y_prob = [], []
    for x, y in loader:
        x = x.to(device)
        with torch.no_grad():
            prob = torch.sigmoid(clf(encoder(x)).squeeze(1))
        y_true.append(y.numpy()); y_prob.append(prob.cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_prob)

# ===============================
# 主函数
# ===============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arff_src_dirs', required=True)
    parser.add_argument('--src_file', default=None, help="Optional single source ARFF file for single-source experiments")
    parser.add_argument('--arff_tgt_file', required=True)
    parser.add_argument('--align_epochs', type=int, default=20)
    parser.add_argument('--final_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_gen_per_seed', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--save_dir', default="./runs/demo_v6")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    set_seed(args.seed)

    # --- 数据加载 ---
    print("📂 Loading data...")
    if args.src_file is not None:
        # 单源模式
        print(f"🎯 Single-source mode: Using only {args.src_file}")
        src_X, src_y = load_arff_dataset(os.path.join(args.arff_src_dirs, args.src_file))
    else:
        # 多源模式（自动排除目标文件）
        exclude_name = os.path.basename(args.arff_tgt_file)
        print(f"🌍 Multi-source mode: Loading all ARFFs except {exclude_name}")
        src_X, src_y = load_multiple_arff_dirs(args.arff_src_dirs, exclude_file=args.arff_tgt_file)

    tgt_X, tgt_y = load_arff_dataset(args.arff_tgt_file)
    print(f"✅ Source dim={src_X.shape[1]}, Target dim={tgt_X.shape[1]}")

    # --- 自动维度映射 ---
    aligner = None
    if src_X.shape[1] != tgt_X.shape[1]:
        aligner = LinearAligner(src_X.shape[1], tgt_X.shape[1]).to(device)
        with torch.no_grad():
            src_X = aligner(torch.tensor(src_X, dtype=torch.float32)).cpu().numpy()
        print(f"🔧 Feature mapped: {src_X.shape[1]} → {tgt_X.shape[1]}")

    # --- 样本权重计算 ---
    pos_ratio = np.sum(tgt_y == 1) / len(tgt_y)
    pos_weight = torch.tensor([(1 - pos_ratio) / pos_ratio], device=device)
    print(f"⚖️ Positive ratio={pos_ratio:.3f}, pos_weight={pos_weight.item():.2f}")

    # --- 数据加载器 ---
    src_loader = DataLoader(TensorDataset(torch.tensor(src_X), torch.tensor(src_y)), batch_size=args.batch_size, shuffle=True)
    tgt_loader = DataLoader(TensorDataset(torch.tensor(tgt_X), torch.tensor(tgt_y)), batch_size=args.batch_size)

    encoder_src = Encoder(tgt_X.shape[1]).to(device)
    encoder_tgt = Encoder(tgt_X.shape[1]).to(device)
    clf = Classifier(64).to(device)
    generator = Generator(16, 1, tgt_X.shape[1]).to(device)

    # --- 域对齐 ---
    print("🔁 Aligning domains...")
    train_alignment(encoder_src, encoder_tgt, torch.tensor(tgt_X, dtype=torch.float32).to(device), device, args.align_epochs)

    # --- 生成样本 ---
    print("🧬 Generating synthetic data...")
    label_set = torch.tensor([0.0, 1.0], device=device)
    n_pos = int(args.n_gen_per_seed * (1 + (1 - pos_ratio)))
    X_syn, y_syn = generate_samples(generator, label_set, n_pos, 16, device)
    all_X = np.concatenate([tgt_X, X_syn.numpy()])
    all_y = np.concatenate([tgt_y, y_syn.numpy().squeeze()])
    print(f"🧩 Synthesized {len(X_syn)} samples, total {len(all_X)}")

    # --- 分类器训练 ---
    aug_loader = DataLoader(TensorDataset(torch.tensor(all_X), torch.tensor(all_y)), batch_size=args.batch_size, shuffle=True)
    opt_clf = torch.optim.Adam(clf.parameters(), lr=5e-4)

    print("🏁 Training classifier...")
    for epoch in range(1, args.final_epochs + 1):
        loss = train_classifier(encoder_tgt, clf, aug_loader, opt_clf, device, pos_weight)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[Epoch {epoch}] Loss={loss:.4f}")

    # --- 测试 ---
    print("📊 Evaluating...")
    y_true, y_prob = predict(encoder_tgt, clf, tgt_loader, device)
    metrics = evaluate(y_true, y_prob)

    print("📈 Final Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # --- 保存结果 ---
    csv_path = os.path.join(args.save_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        for k, v in metrics.items():
            w.writerow([k, f"{v:.4f}"])
    print(f"✅ Results saved to {csv_path}")


if __name__ == "__main__":
    main()