#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_rq2_all_datasets.py
批量运行 RQ2 实验：遍历多个数据集 (AEEEMJIRA, NASA, PROMISE)。
每个数据集下自动跑多个目标项目、多随机种子、多合成样本量。
"""

import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from scipy.stats import wilcoxon

# =====================================
# 全局配置
# =====================================
DATASETS = ["AEEEM", "JIRA", "NASA", "PROMISE"]
ARFF_ROOT = "datasets"
OUT_ROOT = "results_rq2"
SCRIPT = "rq2_cure_diagnostics.py"

SEEDS = [42, 123, 3407, 2024, 999]
N_SYN_LIST = [200, 500, 800]
# =====================================

def run_one_dataset(dataset_name):
    print(f"\n=== 🚀 Running dataset: {dataset_name} ===")
    arff_dir = os.path.join(ARFF_ROOT, dataset_name)
    out_dir = os.path.join(OUT_ROOT, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    targets = sorted([f for f in os.listdir(arff_dir) if f.endswith(".arff")])
    summary_rows = []

    for tgt in tqdm(targets, desc=f"{dataset_name} targets"):
        tgt_path = os.path.join(arff_dir, tgt)
        for seed in SEEDS:
            for n_syn in N_SYN_LIST:
                exp_name = f"{os.path.splitext(tgt)[0]}_seed{seed}_n{n_syn}"
                save_dir = os.path.join(out_dir, exp_name)
                os.makedirs(save_dir, exist_ok=True)

                cmd = [
                    "python", SCRIPT,
                    "--arff_tgt_file", tgt_path,
                    "--save_dir", save_dir,
                    "--n_syn_per_class", str(n_syn),
                    "--seed", str(seed),
                    "--cpu"
                ]
                print("→", " ".join(cmd))
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                csv_path = os.path.join(save_dir, "rq2_diagnostics.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    df["Target"] = os.path.splitext(tgt)[0]
                    df["Seed"] = seed
                    df["n_syn"] = n_syn
                    df["Dataset"] = dataset_name
                    summary_rows.append(df)

    # 汇总
    if not summary_rows:
        print(f"⚠️ {dataset_name}: no results found.")
        return

    all_df = pd.concat(summary_rows, ignore_index=True)
    all_df.to_csv(os.path.join(out_dir, "rq2_all_results.csv"), index=False)

    # Pivot 并计算 Δ
    pivot = all_df.pivot_table(index=["Target", "Seed", "n_syn"],
                               columns="Metric",
                               values=["Before", "After"],
                               aggfunc="mean")
    delta = pivot["After"] - pivot["Before"]
    delta = delta.reset_index()
    delta.to_csv(os.path.join(out_dir, "rq2_delta_summary.csv"), index=False)

    # 计算平均 Δ 与显著性
    metrics = [c for c in delta.columns if c not in ["Target", "Seed", "n_syn"]]
    results = []
    for m in metrics:
        vals = delta[m].dropna()
        if len(vals) > 0:
            stat, p = wilcoxon(vals)
            results.append({
                "Metric": m,
                "MeanΔ": np.mean(vals),
                "MedianΔ": np.median(vals),
                "p(Wilcoxon)": p
            })
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(out_dir, "rq2_significance.csv"), index=False)

    print(f"✅ Finished {dataset_name}, results saved to {out_dir}")

    # 画Δ散点
    metrics_to_plot = ["Coverage@eps", "MMD (latent)", "Brier score", "Margin"]
    for m in metrics_to_plot:
        if m in delta.columns:
            plt.figure(figsize=(6,4))
            for n_syn in N_SYN_LIST:
                vals = delta[delta["n_syn"]==n_syn][m].dropna()
                plt.scatter([n_syn]*len(vals), vals, alpha=0.5, label=f"n={n_syn}" if n_syn==N_SYN_LIST[0] else None)
            plt.axhline(0, color="gray", linestyle="--")
            plt.title(f"{dataset_name}: Δ{m} (After−Before)")
            plt.xlabel("n_syn_per_class")
            plt.ylabel("Δ Value")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"delta_{m.replace(' ','_')}.png"), dpi=200)
            plt.close()

# 主执行循环
if __name__ == "__main__":
    for dataset in DATASETS:
        run_one_dataset(dataset)

    # 汇总所有数据集结果
    all_results = []
    for dataset in DATASETS:
        file_path = os.path.join(OUT_ROOT, dataset, "rq2_delta_summary.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df["Dataset"] = dataset
            all_results.append(df)
    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        df_all.to_csv(os.path.join(OUT_ROOT, "rq2_all_datasets_delta.csv"), index=False)
        print("🌍 Combined summary saved: results_rq2/rq2_all_datasets_delta.csv")