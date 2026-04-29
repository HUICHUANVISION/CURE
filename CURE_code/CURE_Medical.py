#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HIA_Med_Train.py
[Project HIA-Med] 异构医学影像鲁棒迁移框架 - 训练主程序

核心流程：
1. Inputs: 加载预提取的 ResNet 特征向量 (.npy)
2. Phase I: 跨域特征对齐 (MMD Loss)
3. Phase II: 潜在空间特征增强 (Generator)
4. Phase III: 一致性精炼 (Consistency Regularization)
5. Output: 保存训练好的 Encoder 和 Classifier 权重
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, confusion_matrix


# ===============================
# 0. 基础组件与工具
# ===============================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_npy_data(feat_path, label_path):
    """加载 .npy 特征文件 (N, 2048)"""
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Feature file not found: {feat_path}")
    X = np.load(feat_path).astype(np.float32)
    y = np.load(label_path).astype(np.float32)
    # 确保标签是 0/1 (针对二分类任务)
    y = (y > 0).astype(np.float32)
    print(f"✅ Loaded {os.path.basename(feat_path)} | Shape={X.shape}, Pos Ratio={np.mean(y):.2%}")
    return X, y


def evaluate_metrics(y_true, y_prob, threshold=0.5):
    """计算 AUC, F1, Acc 等指标"""
    y_pred = (y_prob >= threshold).astype(int)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    acc = (tp + tn) / len(y_true)
    return {"AUC": auc, "F1": f1, "ACC": acc}


# ===============================
# 1. 网络架构 (HIA-Med Modules)
# ===============================
class HIAEncoder(nn.Module):
    """将 ResNet 的 2048 维特征压缩到潜在空间 (Latent Space)"""

    def __init__(self, input_dim=2048, hidden_dim=512, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()  # Latent Code
        )

    def forward(self, x): return self.net(x)


class HIAClassifier(nn.Module):
    """鲁棒分类器"""

    def __init__(self, input_dim=128, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output Logits
        )

    def forward(self, x): return self.net(x)


class LatentGenerator(nn.Module):
    """条件特征生成器 (用于 Phase II)"""

    def __init__(self, z_dim, label_dim, output_dim):
        super().__init__()
        # Input: Noise + Label -> Latent Feature
        self.net = nn.Sequential(
            nn.Linear(z_dim + label_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, output_dim),
            nn.ReLU()  # 保持与 Encoder 输出空间一致
        )

    def forward(self, z, label):
        return self.net(torch.cat([z, label], dim=1))


# ===============================
# 2. 核心损失函数 (MMD)
# ===============================
class MMDLoss(nn.Module):
    """衡量 Source 和 Target 在 Latent Space 的分布差异"""

    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def gaussian_kernel(self, source, target):
        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = source.size(0)
        kernels = self.gaussian_kernel(source, target)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss


# ===============================
# 3. 三阶段训练流程
# ===============================
def train_phase_1_alignment(encoder, classifier, src_loader, tgt_loader, device, epochs=20):
    """[Phase I] 对齐: Cls Loss + MMD Loss"""
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=1e-3)
    mmd_criterion = MMDLoss()
    bce_criterion = nn.BCEWithLogitsLoss()

    print("\n>>> Phase I: Starting Conditional Alignment...")
    for epoch in range(1, epochs + 1):
        encoder.train();
        classifier.train()
        total_mmd, total_cls = 0, 0
        tgt_iter = iter(tgt_loader)

        for src_x, src_y in src_loader:
            try:
                tgt_x, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_x, _ = next(tgt_iter)

            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x = tgt_x.to(device)

            # Forward
            src_z = encoder(src_x)
            tgt_z = encoder(tgt_x)
            logits = classifier(src_z).squeeze(1)

            loss_cls = bce_criterion(logits, src_y)
            loss_mmd = mmd_criterion(src_z, tgt_z)
            loss = loss_cls + 0.5 * loss_mmd  # 0.5 是 MMD 权重

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_mmd += loss_mmd.item()
            total_cls += loss_cls.item()

        if epoch % 5 == 0:
            print(f"  Epoch {epoch} | Cls: {total_cls:.4f} | MMD: {total_mmd:.4f}")


def generate_latent_samples(generator, n_gen, z_dim, device):
    """[Phase II] 生成: 补充 '恶性' (Label=1) 特征"""
    generator.eval()
    y_tensor = torch.ones((n_gen, 1), dtype=torch.float32, device=device)  # 仅生成正样本
    z_noise = torch.randn(n_gen, z_dim, device=device)
    with torch.no_grad():
        fake_feats = generator(z_noise, y_tensor)
    return fake_feats.cpu(), y_tensor.cpu()


def train_phase_3_refinement(encoder, classifier, combined_loader, device, epochs=30):
    """[Phase III] 精炼: 引入一致性正则化 (Consistency Regularization)"""
    optimizer = torch.optim.Adam(list(classifier.parameters()), lr=5e-4)
    bce_criterion = nn.BCEWithLogitsLoss()

    print("\n>>> Phase III: Classifier Refinement (Consistency Reg)...")
    for epoch in range(1, epochs + 1):
        classifier.train();
        encoder.eval()
        total_loss = 0

        for z, y in combined_loader:
            z, y = z.to(device), y.to(device)

            # 1. Standard Loss
            logits = classifier(z).squeeze(1)
            loss_std = bce_criterion(logits, y)

            # 2. Consistency Reg: 对 Latent 加微小扰动，要求输出一致
            noise = torch.randn_like(z) * 0.05
            logits_pert = classifier(z + noise).squeeze(1)
            loss_cons = F.mse_loss(torch.sigmoid(logits), torch.sigmoid(logits_pert))

            loss = loss_std + 1.0 * loss_cons

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"  Refine Epoch {epoch} | Loss: {total_loss:.4f}")


# ===============================
# 主程序
# ===============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_feats', required=True, help="path to source .npy")
    parser.add_argument('--src_labels', required=True)
    parser.add_argument('--tgt_feats', required=True, help="path to target .npy")
    parser.add_argument('--tgt_labels', required=True)
    parser.add_argument('--save_dir', default="./checkpoints_hia")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(42)

    # 1. 加载数据
    print("📂 Loading Features...")
    src_X, src_y = load_npy_data(args.src_feats, args.src_labels)
    tgt_X, tgt_y = load_npy_data(args.tgt_feats, args.tgt_labels)

    batch_size = 64
    src_loader = DataLoader(TensorDataset(torch.tensor(src_X), torch.tensor(src_y)), batch_size=batch_size,
                            shuffle=True)
    tgt_loader = DataLoader(TensorDataset(torch.tensor(tgt_X), torch.tensor(tgt_y)), batch_size=batch_size,
                            shuffle=True)

    # 2. 初始化模型
    encoder = HIAEncoder(input_dim=src_X.shape[1], latent_dim=128).to(device)
    classifier = HIAClassifier(input_dim=128).to(device)
    generator = LatentGenerator(z_dim=32, label_dim=1, output_dim=128).to(device)

    # 3. Phase I: 对齐
    train_phase_1_alignment(encoder, classifier, src_loader, tgt_loader, device)

    # 4. Phase II: 生成 (补充正样本)
    print("\n>>> Phase II: Augmenting Rare Classes...")
    # 假设补充 500 个正样本
    gen_feats, gen_labels = generate_latent_samples(generator, n_gen=500, z_dim=32, device=device)

    # 准备 Phase III 数据 (Source Latents + Generated Latents)
    encoder.eval()
    with torch.no_grad():
        src_real_z = encoder(torch.tensor(src_X).to(device)).cpu()

    comb_z = torch.cat([src_real_z, gen_feats], dim=0)
    comb_y = torch.cat([torch.tensor(src_y), gen_labels.squeeze()], dim=0)
    comb_loader = DataLoader(TensorDataset(comb_z, comb_y), batch_size=batch_size, shuffle=True)

    # 5. Phase III: 精炼
    train_phase_3_refinement(encoder, classifier, comb_loader, device)

    # 6. 最终测试 (Target Domain)
    print("\n>>> Evaluation on Target Domain...")
    encoder.eval();
    classifier.eval()
    y_true, y_prob = [], []
    test_loader = DataLoader(TensorDataset(torch.tensor(tgt_X), torch.tensor(tgt_y)), batch_size=batch_size)

    for x, y in test_loader:
        x = x.to(device)
        with torch.no_grad():
            prob = torch.sigmoid(classifier(encoder(x))).squeeze(1)
        y_true.append(y.numpy());
        y_prob.append(prob.cpu().numpy())

    metrics = evaluate_metrics(np.concatenate(y_true), np.concatenate(y_prob))
    print(f"🏆 Final Metrics: {metrics}")

    # 7. 保存权重 (供 Step 2 可视化使用)
    torch.save(encoder.state_dict(), os.path.join(args.save_dir, "hia_encoder.pth"))
    torch.save(classifier.state_dict(), os.path.join(args.save_dir, "hia_classifier.pth"))
    print(f"💾 Models saved to {args.save_dir}")


if __name__ == "__main__":
    main()