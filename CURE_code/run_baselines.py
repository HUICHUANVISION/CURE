#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_baselines.py
Baselines for Cross-Project Defect Prediction (CPDP)
Includes oversampling and transfer-learning baselines: SMOTE, ADASYN, TCA, CORAL.
Supports repeated runs (default=30) for statistical stability.
Author: Your Name
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, matthews_corrcoef, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

# ===========================================================
# 工具函数
# ===========================================================
def clean_label(x):
    if isinstance(x, bytes): x = x.decode('utf-8', errors='ignore')
    x = str(x).strip().lower()
    if x in ['y', 'yes', 'bug', 'buggy', 'true', 'defective', '1']:
        return 1
    elif x in ['n', 'no', 'nonbug', 'clean', 'false', 'nondefective', '0']:
        return 0
    try:
        return 1 if float(x) > 0 else 0
    except:
        return 0

def load_arff(file_path):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    label_col = df.columns[-1]
    y = df[label_col].apply(clean_label).astype(np.int32)
    X = df.drop(columns=[label_col]).astype(np.float32)
    return X.values, y.values

def load_arff_dir(folder, exclude_file=None):
    all_X, all_y = [], []
    exclude_name = None
    if exclude_file is not None:
        exclude_name = os.path.basename(exclude_file)
        print(f"🧩 Excluding target file from source: {exclude_name}")

    for f in sorted(os.listdir(folder)):
        if f.endswith(".arff"):
            if exclude_name is not None and f == exclude_name:
                print(f"🚫 Skipping target file {f} from source training set.")
                continue
            X, y = load_arff(os.path.join(folder, f))
            all_X.append(X)
            all_y.append(y)
    return np.vstack(all_X), np.concatenate(all_y)

def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
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

# ===========================================================
# Oversampling Baselines
# ===========================================================
def apply_oversampler(X, y, method, seed):
    method = method.lower()
    np.random.seed(seed)
    if method == "none":
        return X, y
    elif method == "smote":
        sampler = SMOTE(random_state=seed)
    elif method == "adasyn":
        sampler = ADASYN(random_state=seed)
    elif method == "ros":
        sampler = RandomOverSampler(random_state=seed)
    elif method == "borderline":
        sampler = BorderlineSMOTE(random_state=seed)
    elif method == "svmsmote":
        sampler = SVMSMOTE(random_state=seed)
    elif method == "kmeanssmote":
        sampler = KMeansSMOTE(random_state=seed)
    elif method == "smoteenn":
        sampler = SMOTEENN(random_state=seed)
    elif method == "smotetomek":
        sampler = SMOTETomek(random_state=seed)
    else:
        raise ValueError(f"Unknown oversampler: {method}")
    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res

# ===========================================================
# Transfer Baselines
# ===========================================================
def kernel(X1, X2, sigma=1.0):
    X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
    X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
    dist = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
    K = np.exp(-dist / (2 * sigma ** 2))
    return K

def tca(Xs, Xt, dim=50, lamb=1.0, kernel_type='linear', gamma=1.0):
    X = np.vstack((Xs, Xt))
    n, m = X.shape
    ns, nt = Xs.shape[0], Xt.shape[0]
    e = np.vstack((1. / ns * np.ones((ns, 1)), -1. / nt * np.ones((nt, 1))))
    M = e @ e.T
    M = M / np.linalg.norm(M, 'fro')
    H = np.eye(n) - 1. / n * np.ones((n, n))
    if kernel_type == 'linear':
        K = X @ X.T
    else:
        K = kernel(X, X, gamma)
    a = K @ M @ K.T + lamb * np.eye(n)
    b = K @ H @ K.T
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(a).dot(b))
    idx = np.argsort(eigvals)[::-1]
    eigvecs = np.real(eigvecs[:, idx])
    A = eigvecs[:, :dim]
    Z = K @ A
    Z /= np.linalg.norm(Z, axis=0)
    Zs, Zt = Z[:ns, :], Z[ns:, :]
    return Zs, Zt

def coral(Xs, Xt):
    cov_src = np.cov(Xs, rowvar=False) + np.eye(Xs.shape[1])
    cov_tgt = np.cov(Xt, rowvar=False) + np.eye(Xt.shape[1])
    A_coral = np.dot(np.linalg.inv(np.linalg.cholesky(cov_src)),
                     np.linalg.cholesky(cov_tgt))
    Xs_new = np.dot(Xs, A_coral)
    return Xs_new, Xt

# ===========================================================
# 主流程
# ===========================================================
def run_baseline(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # ✅ 单源模式 vs 多源模式
    if args.src_file is not None:
        src_path = os.path.join(args.src_dirs, args.src_file)
        print(f"🎯 Single-source mode: using {src_path}")
        Xs, ys = load_arff(src_path)
    else:
        # 多源模式（自动排除目标文件）
        Xs, ys = load_arff_dir(args.src_dirs, exclude_file=args.tgt_file)

    Xt, yt = load_arff(args.tgt_file)
    print(f"✅ Loaded Source {Xs.shape}, Target {Xt.shape}")

    scaler = StandardScaler().fit(Xs)
    Xs, Xt = scaler.transform(Xs), scaler.transform(Xt)

    methods = [m.strip().lower() for m in args.methods.split(",")]
    all_summary = []

    for method in methods:
        print(f"\n🚀 Running baseline: {method.upper()} for {args.repeats} runs...")
        metrics_list = []
        for r in range(args.repeats):
            seed = 42 + r
            np.random.seed(seed)
            try:
                if method in ["none","smote","adasyn","ros","borderline","svmsmote","kmeanssmote","smoteenn","smotetomek"]:
                    X_train, y_train = apply_oversampler(Xs, ys, method, seed)
                    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=seed)
                    clf.fit(X_train, y_train)
                    y_prob = clf.predict_proba(Xt)[:,1]
                elif method == "tca":
                    Zs, Zt = tca(Xs, Xt, dim=min(50, Xs.shape[1]))
                    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=seed)
                    clf.fit(Zs, ys)
                    y_prob = clf.predict_proba(Zt)[:,1]
                elif method == "coral":
                    Xs_new, Xt_new = coral(Xs, Xt)
                    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=seed)
                    clf.fit(Xs_new, ys)
                    y_prob = clf.predict_proba(Xt_new)[:,1]
                else:
                    raise ValueError(f"Unknown method: {method}")

                metrics = evaluate(yt, y_prob)
                metrics["Run"] = r + 1
                metrics["Method"] = method.upper()
                metrics_list.append(metrics)
            except Exception as e:
                print(f"❌ {method.upper()} run {r+1} failed: {e}")

        # 汇总平均 & 标准差
        df_runs = pd.DataFrame(metrics_list)
        mean_metrics = df_runs.mean(numeric_only=True)
        std_metrics = df_runs.std(numeric_only=True)
        summary = {f"{k}_mean": mean_metrics[k] for k in mean_metrics.keys()}
        summary.update({f"{k}_std": std_metrics[k] for k in std_metrics.keys()})
        summary["Method"] = method.upper()
        all_summary.append(summary)

        # 保存每次运行
        df_runs.to_csv(os.path.join(args.save_dir, f"{method}_runs.csv"), index=False)

    # 保存总体统计
    df_summary = pd.DataFrame(all_summary)
    df_summary.to_csv(os.path.join(args.save_dir, "baseline_summary.csv"), index=False)
    print("\n✅ Finished all baselines. Summary saved to baseline_summary.csv")

# ===========================================================
# 命令行入口
# ===========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CPDP baselines with multiple repeats")
    parser.add_argument("--src_file", default=None, help="Specify one source ARFF file for single-source experiment")
    parser.add_argument("--src_dirs", required=True, help="Source ARFF directory")
    parser.add_argument("--tgt_file", required=True, help="Target ARFF file")
    parser.add_argument("--methods", type=str, default="none,smote,adasyn,ros,tca,coral",
                        help="Comma-separated baseline methods")
    parser.add_argument("--save_dir", default="./results/baselines", help="Directory to save results")
    parser.add_argument("--repeats", type=int, default=30, help="Number of repeated runs (default=30)")
    parser.add_argument("--cpu", action="store_true", help="Dummy flag for compatibility")
    args = parser.parse_args()
    run_baseline(args)