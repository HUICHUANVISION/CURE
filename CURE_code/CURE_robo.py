#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HIA_Robo_Train.py
[Project HIA-Robo] Sim-to-Real 鲁棒感知迁移框架 - 训练主程序

核心流程 (Sim-to-Real Pipeline):
1. Inputs: 加载离线提取的感知特征 (ResNet Feature Vectors from Sim/Real)
2. Phase I: 仿真-现实特征对齐 (Sim-to-Real Alignment via MMD)
3. Phase II: 危险场景特征生成 (Collision Scenario Generation)
4. Phase III: 安全性精炼 (Safety-Critical Refinement)
5. Output: 导出用于部署的 Encoder 和 Classifier 权重

Target: 用于机器人地形分类 / 可通行性分析 (Traversability Analysis)
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


def load_perception_feats(feat_path, label_path):
    """
    加载 .npy 感知特征文件 (N, 2048)
    Label Definition: 0 = Free Space (可通行), 1 = Obstacle (障碍物/不可通行)
    """
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Feature file not found: {feat_path}")
    X = np.load(feat_path).astype(np.float32)
    y = np.load(label_path).astype(np.float32)

    # 确保标签二值化 (针对障碍物检测)
    y = (y > 0).astype(np.float32)

    # 计算障碍物占比
    obs_ratio = np.mean(y)
    print(f"✅ [Data Loaded] {os.path.basename(feat_path)} | Count={len(X)} | Obstacle Ratio={obs_ratio:.2%}")
    return X, y


def evaluate_safety_metrics(y_true, y_prob, threshold=0.5):
    """
    计算机器人安全关键指标
    - FPR (Ghost Braking Rate): 误报率，把路面当障碍物，导致幽灵刹车。
    - FNR (Collision Risk): 漏报率，没看见障碍物，导致碰撞。
    """
    y_pred = (y_prob >= threshold).astype(int)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # 常用指标
    acc = (tp + tn) / len(y_true)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    # 机器人核心安全指标
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0  # 幽灵刹车风险
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0  # 碰撞风险

    return {
        "AUC": auc,
        "Acc": acc,
        "F1": f1,
        "FPR (Ghost)": fpr,
        "FNR (Collision)": fnr
    }


# ===============================
# 1. 网络架构 (Perception Modules)
# ===============================
class RoboEncoder(nn.Module):
    """
    特征压缩器: 将 ResNet 的高维视觉特征映射到语义潜在空间
    Input: 2048-dim (Visual Feats) -> Output: 128-dim (Semantic Latent)
    """

    def __init__(self, input_dim=2048, hidden_dim=512, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )

    def forward(self, x): return self.net(x)


class TraversabilityClassifier(nn.Module):
    """
    可通行性分类器
    Output: P(Obstacle), 值越大代表风险越高
    """

    def __init__(self, input_dim=128, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Logits
        )

    def forward(self, x): return self.net(x)


class ScenarioGenerator(nn.Module):
    """
    虚拟场景生成器 (Phase II)
    在特征空间生成长尾/危险场景的特征 (Corner Cases)
    """

    def __init__(self, z_dim, label_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + label_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )

    def forward(self, z, label):
        return self.net(torch.cat([z, label], dim=1))


# ===============================
# 2. 核心损失函数 (Domain Adaptation)
# ===============================
class MMDLoss(nn.Module):
    """
    Sim-to-Real Alignment Loss
    计算仿真数据 (Sim) 和真实数据 (Real) 在特征空间的分布距离
    """

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
# 3. 训练流程 (Sim-to-Real Pipeline)
# ===============================
def train_sim2real_alignment(encoder, classifier, sim_loader, real_loader, device, epochs=20):
    """[Phase I] Sim-to-Real Feature Alignment"""
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=1e-3)
    mmd_criterion = MMDLoss()
    bce_criterion = nn.BCEWithLogitsLoss()

    print("\n>>> 🚀 Phase I: Starting Sim-to-Real Alignment...")
    for epoch in range(1, epochs + 1):
        encoder.train();
        classifier.train()
        total_mmd, total_cls = 0, 0
        real_iter = iter(real_loader)

        for sim_x, sim_y in sim_loader:
            try:
                real_x, _ = next(real_iter)
            except StopIteration:
                real_iter = iter(real_loader)
                real_x, _ = next(real_iter)

            sim_x, sim_y = sim_x.to(device), sim_y.to(device)
            real_x = real_x.to(device)  # Real world data has no labels during alignment

            # Forward
            sim_z = encoder(sim_x)
            real_z = encoder(real_x)
            logits = classifier(sim_z).squeeze(1)

            loss_cls = bce_criterion(logits, sim_y)
            loss_mmd = mmd_criterion(sim_z, real_z)

            # Loss = Perception Accuracy (Sim) + Domain Alignment (Sim vs Real)
            loss = loss_cls + 0.5 * loss_mmd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_mmd += loss_mmd.item()
            total_cls += loss_cls.item()

        if epoch % 5 == 0:
            print(f"  [Epoch {epoch}] Sim Acc Loss: {total_cls:.4f} | Domain Gap (MMD): {total_mmd:.4f}")


def generate_virtual_obstacles(generator, n_gen, z_dim, device):
    """[Phase II] 生成虚拟障碍物特征 (Corner Cases)"""
    generator.eval()
    # 强制生成 Label=1 (障碍物/碰撞风险)
    y_tensor = torch.ones((n_gen, 1), dtype=torch.float32, device=device)
    z_noise = torch.randn(n_gen, z_dim, device=device)
    with torch.no_grad():
        fake_feats = generator(z_noise, y_tensor)
    return fake_feats.cpu(), y_tensor.cpu()


def train_safety_refinement(encoder, classifier, combined_loader, device, epochs=30):
    """[Phase III] 安全性精炼 (Consistency Reg)"""
    optimizer = torch.optim.Adam(list(classifier.parameters()), lr=5e-4)
    bce_criterion = nn.BCEWithLogitsLoss()

    print("\n>>> 🛡️ Phase III: Safety-Critical Refinement (Consistency)...")
    for epoch in range(1, epochs + 1):
        classifier.train();
        encoder.eval()
        total_loss = 0

        for z, y in combined_loader:
            z, y = z.to(device), y.to(device)

            # 1. Standard Perception Loss
            logits = classifier(z).squeeze(1)
            loss_std = bce_criterion(logits, y)

            # 2. Consistency Regularization (模拟传感器噪声)
            # 确保输入有微小扰动时，机器人的判断保持一致
            noise = torch.randn_like(z) * 0.05
            logits_pert = classifier(z + noise).squeeze(1)
            loss_cons = F.mse_loss(torch.sigmoid(logits), torch.sigmoid(logits_pert))

            loss = loss_std + 1.0 * loss_cons

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"  [Refine Epoch {epoch}] Loss: {total_loss:.4f}")


# ===============================
# 主程序
# ===============================
def main():
    parser = argparse.ArgumentParser(description="HIA-Robo: Sim-to-Real Training Node")
    parser.add_argument('--sim_feats', required=True, help="Path to Simulation features (Source, e.g., GTAV)")
    parser.add_argument('--sim_labels', required=True)
    parser.add_argument('--real_feats', required=True, help="Path to Real-world features (Target, e.g., Cityscapes)")
    parser.add_argument('--real_labels', required=True, help="Real labels (Only for evaluation, not used in training)")
    parser.add_argument('--save_dir', default="./checkpoints_robo", help="Directory to export ROS models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(42)

    # 1. 加载数据
    print("🤖 Initializing HIA-Robo Node...")
    sim_X, sim_y = load_perception_feats(args.sim_feats, args.sim_labels)
    real_X, real_y = load_perception_feats(args.real_feats, args.real_labels)

    batch_size = 64
    sim_loader = DataLoader(TensorDataset(torch.tensor(sim_X), torch.tensor(sim_y)), batch_size=batch_size,
                            shuffle=True)
    real_loader = DataLoader(TensorDataset(torch.tensor(real_X), torch.tensor(real_y)), batch_size=batch_size,
                             shuffle=True)

    # 2. 初始化感知模型
    encoder = RoboEncoder(input_dim=sim_X.shape[1], latent_dim=128).to(device)
    classifier = TraversabilityClassifier(input_dim=128).to(device)
    generator = ScenarioGenerator(z_dim=32, label_dim=1, output_dim=128).to(device)

    # 3. Phase I: Sim-to-Real Alignment
    train_sim2real_alignment(encoder, classifier, sim_loader, real_loader, device)

    # 4. Phase II: Generating Virtual Collision Risks
    print("\n>>> ⚠️ Phase II: Generating Virtual Corner Cases...")
    # 模拟生成 500 个极端障碍物样本
    gen_feats, gen_labels = generate_virtual_obstacles(generator, n_gen=500, z_dim=32, device=device)

    # 混合数据: 真实仿真特征 + 虚拟生成的危险特征
    encoder.eval()
    with torch.no_grad():
        sim_real_z = encoder(torch.tensor(sim_X).to(device)).cpu()

    comb_z = torch.cat([sim_real_z, gen_feats], dim=0)
    comb_y = torch.cat([torch.tensor(sim_y), gen_labels.squeeze()], dim=0)
    comb_loader = DataLoader(TensorDataset(comb_z, comb_y), batch_size=batch_size, shuffle=True)

    # 5. Phase III: Safety Refinement
    train_safety_refinement(encoder, classifier, comb_loader, device)

    # 6. 最终实车验证 (Evaluation on Real World Data)
    print("\n>>> 🏁 Evaluating on Real World Data (Cityscapes)...")
    encoder.eval();
    classifier.eval()
    y_true, y_prob = [], []
    test_loader = DataLoader(TensorDataset(torch.tensor(real_X), torch.tensor(real_y)), batch_size=batch_size)

    for x, y in test_loader:
        x = x.to(device)
        with torch.no_grad():
            # Output represents P(Obstacle)
            prob = torch.sigmoid(classifier(encoder(x))).squeeze(1)
        y_true.append(y.numpy())
        y_prob.append(prob.cpu().numpy())

    metrics = evaluate_safety_metrics(np.concatenate(y_true), np.concatenate(y_prob))
    print(f"🏆 [Real-World Performance] {metrics}")

    # 7. 导出模型 (For ROS Integration)
    torch.save(encoder.state_dict(), os.path.join(args.save_dir, "robo_encoder.pth"))
    torch.save(classifier.state_dict(), os.path.join(args.save_dir, "robo_classifier.pth"))
    print(f"💾 Models exported to {args.save_dir} (Ready for ROS Node)")


if __name__ == "__main__":
    main()