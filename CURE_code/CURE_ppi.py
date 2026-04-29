import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score

# ================= 配置参数 =================
CONFIG = {
    'input_dim': 512,  # HGNN Cross-node layer 输出维度
    'latent_dim': 128,  # 对齐空间的维度
    'hidden_dim': 256,  # 中间层维度
    'batch_size': 256,  # PPI数据量大，Batch大一点稳
    'lr': 1e-4,  # 学习率
    'epochs_stage1': 20,  # 对齐阶段 Epochs
    'epochs_stage3': 30,  # 精炼阶段 Epochs
    'mmd_weight': 0.5,  # MMD 对齐损失权重
    'gen_weight': 0.1,  # 生成损失权重
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


# ================= 核心模块 =================

class FeatureProjector(nn.Module):
    """
    [Encoder] 将不同物种的特征投影到公共子空间
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CONFIG['input_dim'], CONFIG['hidden_dim']),
            nn.BatchNorm1d(CONFIG['hidden_dim']),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),  # 防止过拟合特定物种
            nn.Linear(CONFIG['hidden_dim'], CONFIG['latent_dim']),
            nn.BatchNorm1d(CONFIG['latent_dim']),
            nn.Tanh()  # 将特征压缩到 [-1, 1]，方便 MMD 对齐
        )

    def forward(self, x):
        return self.net(x)


class InteractionPredictor(nn.Module):
    """
    [Classifier] 预测互作概率
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(CONFIG['latent_dim'], 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出 Logits
        )

    def forward(self, x):
        return self.net(x)


class LatentGenerator(nn.Module):
    """
    [Generator] 生成潜在的互作特征
    """

    def __init__(self):
        super().__init__()
        # Input: Noise (32) + Label (1)
        self.net = nn.Sequential(
            nn.Linear(32 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, CONFIG['latent_dim']),
            nn.Tanh()  # 输出范围需与 Encoder 一致
        )

    def forward(self, z, label):
        x = torch.cat([z, label], dim=1)
        return self.net(x)


# ================= 辅助函数 =================

def mmd_loss(source, target):
    """
    计算两个分布的 MMD 距离 (RBF Kernel)
    """

    def gaussian_kernel(x, y, sigma=1.0):
        # 简化版 RBF
        beta = 1. / (2. * sigma)
        dist = torch.cdist(x, y) ** 2
        return torch.exp(-beta * dist).mean()

    xx = gaussian_kernel(source, source)
    yy = gaussian_kernel(target, target)
    xy = gaussian_kernel(source, target)
    return xx + yy - 2 * xy


def evaluate(encoder, classifier, loader, name="Test"):
    encoder.eval();
    classifier.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(CONFIG['device'])
            z = encoder(x)
            logits = classifier(z)
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    auprc = average_precision_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_preds)

    print(f"📊 [{name}] AUPRC: {auprc:.4f} | AUROC: {auroc:.4f}")
    return auprc


# ================= 训练逻辑 =================

def train_hia_ppi():
    # 1. 模拟数据加载 (请替换为真实的 load_npy)
    # 假设我们已经有了 human_feats 和 mouse_feats
    print("🚀 开始 HIA-PPI 训练流程...")

    # 初始化网络
    encoder = FeatureProjector().to(CONFIG['device'])
    classifier = InteractionPredictor().to(CONFIG['device'])
    generator = LatentGenerator().to(CONFIG['device'])

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=CONFIG['lr']
    )

    # ---------------- Phase I: 对齐 (Alignment) ----------------
    print("\n>>> Phase I: 跨物种特征对齐 (Human <-> Target)")
    for epoch in range(CONFIG['epochs_stage1']):
        encoder.train();
        classifier.train()
        # 伪代码：从 Dataloader 取数据
        # human_x, human_y = next(human_loader)
        # target_x, _ = next(target_loader)

        # 模拟数据
        human_x = torch.randn(64, 512).to(CONFIG['device'])
        human_y = torch.randint(0, 2, (64, 1)).float().to(CONFIG['device'])
        target_x = torch.randn(64, 512).to(CONFIG['device'])  # Target 无标签

        # Forward
        z_human = encoder(human_x)
        z_target = encoder(target_x)

        # Loss 1: Human 上的分类准确性
        pred_human = classifier(z_human)
        loss_cls = F.binary_cross_entropy_with_logits(pred_human, human_y)

        # Loss 2: Human 和 Target 的分布距离
        loss_mmd = mmd_loss(z_human, z_target)

        loss = loss_cls + CONFIG['mmd_weight'] * loss_mmd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Cls Loss={loss_cls.item():.4f}, MMD Loss={loss_mmd.item():.4f}")

    # ---------------- Phase II & III: 生成与精炼 ----------------
    print("\n>>> Phase III: 生成增强与一致性精炼")
    # 这一步我们将 生成器 和 分类器 联合训练
    # 也就是 CURE 论文中的 Phase II + III 融合

    opt_gen = torch.optim.Adam(generator.parameters(), lr=CONFIG['lr'])

    for epoch in range(CONFIG['epochs_stage3']):
        # 1. 生成虚拟 Positive 样本 (补充稀缺的互作数据)
        z_noise = torch.randn(64, 32).to(CONFIG['device'])
        label_pos = torch.ones(64, 1).to(CONFIG['device'])  # 强制生成 Positive
        z_gen = generator(z_noise, label_pos)

        # 2. 混合真实数据和生成数据
        # real_z = encoder(human_x)
        # z_combined = torch.cat([real_z, z_gen])

        # 3. 一致性损失 (Consistency Loss)
        # 对 Target 数据加微小扰动，预测结果不应改变
        target_x = torch.randn(64, 512).to(CONFIG['device'])
        z_target = encoder(target_x)

        noise = torch.randn_like(z_target) * 0.05
        logits_clean = classifier(z_target)
        logits_noisy = classifier(z_target + noise)

        loss_cons = F.mse_loss(torch.sigmoid(logits_clean), torch.sigmoid(logits_noisy))

        # 4. 生成器的损失 (让生成的样本能够欺骗分类器 - 类似 GAN 但更简单)
        # 我们希望生成的 z_gen 被分类器认为是 Positive
        pred_gen = classifier(z_gen)
        loss_gen = F.binary_cross_entropy_with_logits(pred_gen, label_pos)

        # 总损失更新
        loss_total = loss_cons + 0.1 * loss_gen

        optimizer.zero_grad()
        opt_gen.zero_grad()
        loss_total.backward()
        optimizer.step()
        opt_gen.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Cons Loss={loss_cons.item():.4f}, Gen Loss={loss_gen.item():.4f}")

    print("\n✅ 训练完成，模型已准备好进行跨物种预测。")


if __name__ == '__main__':
    train_hia_ppi()