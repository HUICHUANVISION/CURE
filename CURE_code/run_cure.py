#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_cure.py
Repeated experiment runner for CURE (Conditional Unaligned Representation Enhancement)
Supports both multi-source and single-source CPDP experiments.
Ensures output structure matches baseline experiments.
"""

import os
import argparse
import numpy as np
import pandas as pd
import subprocess
from tqdm import trange


# ===========================================================
# 执行单次 CURE 运行
# ===========================================================
def run_single(args, seed, run_id):
    """调用一次 CURE_full_auto_label_v6.py"""
    save_dir = os.path.join(args.save_dir, f"run_{run_id}")
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        "python", "CURE.py",
        "--arff_src_dirs", args.arff_src_dirs,
        "--arff_tgt_file", args.arff_tgt_file,
        "--align_epochs", str(args.align_epochs),
        "--final_epochs", str(args.final_epochs),
        "--batch_size", str(args.batch_size),
        "--n_gen_per_seed", str(args.n_gen_per_seed),
        "--seed", str(seed),
        "--save_dir", save_dir
    ]

    # 单源模式支持
    if args.src_file:
        cmd += ["--src_file", args.src_file]

    if args.cpu:
        cmd.append("--cpu")

    print(f"\n🚀 [Run {run_id}] Seed={seed}")
    subprocess.run(cmd, check=True)
    return os.path.join(save_dir, "results.csv")


# ===========================================================
# 汇总多个运行的结果
# ===========================================================
def collect_results(run_dirs):
    """汇总多个 run 的 results.csv"""
    records = []
    for csv_path in run_dirs:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            metrics = dict(zip(df["Metric"], df["Value"].astype(float)))
            records.append(metrics)
        else:
            print(f"⚠️ Missing result: {csv_path}")
    df_all = pd.DataFrame(records)
    if df_all.empty:
        raise ValueError("No valid results found!")
    mean = df_all.mean(numeric_only=True)
    std = df_all.std(numeric_only=True)
    summary = pd.DataFrame({
        "Metric": mean.index,
        "Mean": mean.values,
        "Std": std.values
    })
    return df_all, summary


# ===========================================================
# 主流程
# ===========================================================
def main():
    parser = argparse.ArgumentParser(description="Run CURE multiple times (single/multi-source)")
    parser.add_argument("--arff_src_dirs", required=True, help="Source directory containing .arff files")
    parser.add_argument("--arff_tgt_file", required=True, help="Target ARFF file path")
    parser.add_argument("--src_file", default=None, help="Specify single source file for single-source experiments")
    parser.add_argument("--align_epochs", type=int, default=20)
    parser.add_argument("--final_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_gen_per_seed", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=30)  # ✅ 关键参数
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save_dir", default="./results/CURE_runs", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    csv_paths = []

    # =========================
    # Repeated Runs
    # =========================
    for i in trange(args.repeats, desc="Running CURE"):
        seed = args.seed + i
        csv_path = run_single(args, seed, i + 1)
        csv_paths.append(csv_path)

    # =========================
    # Aggregation
    # =========================
    print("\n📊 Aggregating all runs...")
    df_all, df_summary = collect_results(csv_paths)

    df_all.to_csv(os.path.join(args.save_dir, "cure_all_runs.csv"), index=False)
    df_summary.to_csv(os.path.join(args.save_dir, "cure_summary.csv"), index=False)

    print(f"✅ Saved all results to:")
    print(f"   - cure_all_runs.csv")
    print(f"   - cure_summary.csv")


if __name__ == "__main__":
    main()