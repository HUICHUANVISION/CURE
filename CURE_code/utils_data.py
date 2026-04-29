# -*- coding: utf-8 -*-
"""
utils_data.py —— HDP实验通用数据加载工具（改进版）
修复:
- 非数值特征自动编码
- 自动检测标签列
- 健壮标签清洗
"""

import os
import numpy as np
import pandas as pd
from scipy.io import arff


# ==========================================================
# 标签清洗函数
# ==========================================================
def clean_label(x):
    """统一清洗缺陷标签"""
    if isinstance(x, bytes):
        x = x.decode('utf-8', errors='ignore')
    x = str(x).strip().lower()

    pos = {
        '1', '1.0', 'y', 'yes', 't', 'true', 'bug', 'buggy',
        'defective', 'defect', 'faulty', 'positive'
    }
    neg = {
        '0', '0.0', 'n', 'no', 'f', 'false', 'clean',
        'nonbug', 'nondefective', 'nondefect', 'ok', 'negative'
    }

    if x in pos:
        return 1
    elif x in neg:
        return 0

    try:
        return 1 if float(x) > 0 else 0
    except Exception:
        pass

    if any(k in x for k in ['bug', 'defect', 'true', 'fault']):
        return 1
    return 0


# ==========================================================
# 单文件加载
# ==========================================================
def load_arff(file_path):
    """
    加载单个 .arff 文件
    自动识别标签列，并清洗标签为 {0,1}
    自动编码非数值特征
    """
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)

    # 自动识别标签列
    label_candidates = [
        c for c in df.columns
        if any(k in c.lower() for k in ['bug', 'defect', 'label', 'target'])
    ]
    label_col = label_candidates[-1] if label_candidates else df.columns[-1]

    # 清洗标签
    y = df[label_col].apply(clean_label).astype(np.int32)

    # 特征处理：去掉标签列
    X = df.drop(columns=[label_col]).copy()

    # 自动编码非数值特征
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c], _ = pd.factorize(X[c])
    X = X.astype(np.float32)

    # 替换异常
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print(f"📂 Loaded {os.path.basename(file_path)} | X:{X.shape} | y:{np.unique(y, return_counts=True)}")
    return X.values, y.values


# ==========================================================
# 文件夹加载（仅用于同构数据集）
# ==========================================================
def load_arff_dir(folder, exclude_file=None):
    all_X, all_y = [], []
    exclude_name = os.path.basename(exclude_file) if exclude_file else None

    print(f"📁 Loading ARFF dir: {folder}")
    if exclude_name:
        print(f"🧩 Excluding target file from source: {exclude_name}")

    files = sorted(f for f in os.listdir(folder) if f.endswith(".arff"))
    for f in files:
        if exclude_name and f == exclude_name:
            print(f"🚫 Skipping target file {f} from source training set.")
            continue

        fp = os.path.join(folder, f)
        try:
            X, y = load_arff(fp)
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            print(f"⚠️ Failed to load {f}: {e}")

    if len(all_X) == 0:
        raise ValueError("No valid ARFF files loaded.")

    dim_set = set(x.shape[1] for x in all_X)
    if len(dim_set) > 1:
        print(f"⚠️ Warning: Feature dimensions differ across files in {folder}: {dim_set}")
        min_dim = min(dim_set)
        all_X = [x[:, :min_dim] for x in all_X]

    return np.vstack(all_X), np.concatenate(all_y)