#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CURE_RQ3_ablation.py
Ablation study for CURE components: Domain Alignment (DA), Class-Conditional Sampling (CS),
and Weight Smoothing (WS).
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

# ====================================================
# 工具函数
# ====================================================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def clean_and_binarize_label(x):
    if isinstance(x, bytes):
        x = x.decode('utf-8', errors='ignore')
    x = str(x).strip().lower()
    if x in ['y','yes','bug','buggy','true','defective','1']: return 1
    if x in ['n','no','clean','false','nondefective','0']: return 0
    try: return 1 if float(x)>0 else 0
    except: return 0

def load_arff_dataset(file_path):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    y = df.iloc[:, -1].apply(clean_and_binarize_label).astype(np.float32).values
    X = df.iloc[:, :-1].astype(np.float32).values
    print(f"✅ Loaded {os.path.basename(file_path)} | Shape={df.shape}")
    return X, y

def load_multiple_arff_dirs(arff_dir, exclude_file=None):
    all_X, all_y = [], []
    exclude = os.path.basename(exclude_file) if exclude_file else None
    for f in sorted(os.listdir(arff_dir)):
        if f.endswith('.arff') and f != exclude:
            X, y = load_arff_dataset(os.path.join(arff_dir, f))
            all_X.append(X); all_y.append(y)
    return np.vstack(all_X), np.concatenate(all_y)

def evaluate(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    if len(np.unique(y_true))<2 or len(np.unique(y_pred))<2:
        return {k:0.0 for k in ["AUC","F1","Recall","Precision","MCC","Pd","Pf","GM"]}
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    pd_ = tp/(tp+fn) if tp+fn>0 else 0
    pf_ = fp/(fp+tn) if fp+tn>0 else 0
    gm = np.sqrt(pd_*(1-pf_))
    return {"AUC": auc, "F1": f1, "Recall": recall, "Precision": precision,
            "MCC": mcc, "Pd": pd_, "Pf": pf_, "GM": gm}

# ====================================================
# 模型模块
# ====================================================
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

# ====================================================
# MMD Loss
# ====================================================
class MMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
    def gaussian_kernel(self, s, t):
        total = torch.cat([s, t], dim=0)
        L2 = ((total.unsqueeze(0)-total.unsqueeze(1))**2).sum(2)
        bw = torch.mean(L2).detach() / (self.kernel_mul ** (self.kernel_num//2))
        bws = [bw*(self.kernel_mul**i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2/bw_) for bw_ in bws]
        return sum(kernel_val)
    def forward(self, s, t):
        bs = s.size(0)
        kernels = self.gaussian_kernel(s, t)
        XX, YY = kernels[:bs,:bs], kernels[bs:,bs:]
        XY, YX = kernels[:bs,bs:], kernels[bs:,:bs]
        return torch.mean(XX + YY - XY - YX)

# ====================================================
# Train / Predict
# ====================================================
def train_alignment(encoder_src, encoder_tgt, tgt_X, device, epochs=10):
    optimizer = torch.optim.Adam(list(encoder_src.parameters())+list(encoder_tgt.parameters()), lr=1e-3)
    mmd = MMDLoss()
    src_sim = torch.randn(tgt_X.size(0), encoder_src.net[0].in_features, device=device)
    for e in range(epochs):
        optimizer.zero_grad()
        loss = mmd(encoder_src(src_sim), encoder_tgt(tgt_X))
        loss.backward(); optimizer.step()
    print(f"✅ MMD alignment done (final loss={loss.item():.4f})")

def generate_samples(generator, label_set, n_gen, z_dim, device, conditional=True):
    generator.eval(); Xs, Ys = [], []
    if conditional:
        for y in label_set:
            y_tensor = torch.full((n_gen, 1), y, dtype=torch.float32, device=device)
            z = torch.randn(n_gen, z_dim, device=device)
            fake = generator(z, y_tensor)
            Xs.append(fake.detach().cpu()); Ys.append(y_tensor.cpu())
    else:
        z = torch.randn(n_gen*2, z_dim, device=device)
        y_rand = torch.randint(0,2,(n_gen*2,1),dtype=torch.float32,device=device)
        fake = generator(z, y_rand)
        Xs.append(fake.detach().cpu()); Ys.append(y_rand.cpu())
    return torch.cat(Xs,0), torch.cat(Ys,0)

def train_classifier(encoder, clf, loader, optimizer, device, pos_weight=None, ws=False):
    encoder.eval(); clf.train(); total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad(): z = encoder(x)
        logits = clf(z).squeeze(1)
        if ws:
            eps = 0.05
            y_s = y*(1-eps) + 0.5*eps
            bce = F.binary_cross_entropy_with_logits(logits, y_s, pos_weight=pos_weight)
            p = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
            entropy_penalty = (p*torch.log(p) + (1-p)*torch.log(1-p)).mean()
            loss = bce - 0.01 * entropy_penalty
        else:
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

# ====================================================
# 主流程
# ====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arff_src_dirs', required=True)
    parser.add_argument('--arff_tgt_file', required=True)
    parser.add_argument('--da', default='mmd', choices=['none','mmd'])
    parser.add_argument('--cs', action='store_true', help="Enable class-conditional sampling")
    parser.add_argument('--ws', action='store_true', help="Enable weight smoothing and regularization")
    parser.add_argument('--n_gen_per_seed', type=int, default=200)
    parser.add_argument('--align_epochs', type=int, default=20)
    parser.add_argument('--final_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--save_dir', default="./runs/RQ3")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    set_seed(args.seed)

    # 数据加载
    print("📂 Loading data...")
    src_X, src_y = load_multiple_arff_dirs(args.arff_src_dirs, exclude_file=args.arff_tgt_file)
    tgt_X, tgt_y = load_arff_dataset(args.arff_tgt_file)
    pos_ratio = np.sum(tgt_y==1)/len(tgt_y)
    pos_weight = torch.tensor([(1-pos_ratio)/pos_ratio], device=device)

    src_loader = DataLoader(TensorDataset(torch.tensor(src_X), torch.tensor(src_y)), batch_size=args.batch_size, shuffle=True)
    tgt_loader = DataLoader(TensorDataset(torch.tensor(tgt_X), torch.tensor(tgt_y)), batch_size=args.batch_size)

    encoder_src = Encoder(tgt_X.shape[1]).to(device)
    encoder_tgt = Encoder(tgt_X.shape[1]).to(device)
    clf = Classifier(64).to(device)
    generator = Generator(16, 1, tgt_X.shape[1]).to(device)

    # 域对齐
    if args.da == "mmd":
        print("🔁 Domain alignment: MMD")
        train_alignment(encoder_src, encoder_tgt, torch.tensor(tgt_X, dtype=torch.float32).to(device), device, args.align_epochs)
    else:
        print("🚫 Domain alignment disabled")

    # 生成样本
    print(f"🧬 Generating samples (CS={args.cs}) ...")
    label_set = torch.tensor([0.0,1.0], device=device)
    X_syn, y_syn = generate_samples(generator, label_set, args.n_gen_per_seed, 16, device, conditional=args.cs)
    all_X = np.concatenate([tgt_X, X_syn.numpy()])
    all_y = np.concatenate([tgt_y, y_syn.numpy().squeeze()])

    # 训练分类器
    aug_loader = DataLoader(TensorDataset(torch.tensor(all_X), torch.tensor(all_y)), batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.Adam(clf.parameters(), lr=5e-4)
    print(f"🏁 Training classifier (WS={args.ws})...")
    for epoch in range(1, args.final_epochs+1):
        loss = train_classifier(encoder_tgt, clf, aug_loader, opt, device, pos_weight, ws=args.ws)
        if epoch%10==0 or epoch==1: print(f"[Epoch {epoch}] loss={loss:.4f}")

    # 测试
    y_true, y_prob = predict(encoder_tgt, clf, tgt_loader, device)
    metrics = evaluate(y_true, y_prob)
    print("📊 Final Metrics:")
    for k,v in metrics.items(): print(f"{k}: {v:.4f}")

    # 保存结果
    fname = f"results_DA={args.da}_CS={int(args.cs)}_WS={int(args.ws)}_seed={args.seed}.csv"
    csv_path = os.path.join(args.save_dir, fname)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric","Value"])
        for k,v in metrics.items(): w.writerow([k, f"{v:.4f}"])
    print(f"✅ Results saved to {csv_path}")

if __name__ == "__main__":
    main()