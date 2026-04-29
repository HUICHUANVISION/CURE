#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_full_experiment_suite.py
============================
自动执行所有跨项目缺陷预测实验：
- HDP（异质跨项目）
- IDP（同质跨项目）
每个环境包含：
  - 单源（Single-Source）
  - 多源（Multi-Source）
每组实验运行：
  - Baseline (run_baselines.py)
  - CURE (run_cure.py)
每次实验重复 30 次，并在最后输出汇总 summary_all.csv
"""

import os
import subprocess
import pandas as pd
from datetime import datetime

# ===========================================================
# 全局配置
# ===========================================================
REPEATS = 30
PYTHON = "python"  # 你的环境默认Python解释器
DATA_ROOT = "/Users/ding/PycharmProjects/TransDefect/Datasets"
RESULT_ROOT = "./results_suite"

os.makedirs(RESULT_ROOT, exist_ok=True)

# ===========================================================
# 通用执行函数
# ===========================================================
def run_experiment(script, args_dict, tag):
    """统一运行命令行子进程"""
    cmd = [PYTHON, script]
    for k, v in args_dict.items():
        if isinstance(v, bool):
            if v: cmd.append(f"--{k}")
        else:
            cmd += [f"--{k}", str(v)]
    print(f"\n🚀 Running [{tag}] ...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"✅ Finished: {tag}\n")

# ===========================================================
# 结果汇总函数
# ===========================================================
def collect_all_results(root_dir):
    """扫描所有子目录中的 cure_summary.csv / baseline_summary.csv"""
    rows = []
    for subdir, _, files in os.walk(root_dir):
        for fname in files:
            if fname.endswith("_summary.csv"):
                fpath = os.path.join(subdir, fname)
                df = pd.read_csv(fpath)
                if "Mean" in df.columns:
                    summary = {m: v for m, v in zip(df["Metric"], df["Mean"])}
                    summary["Std"] = df["Std"].mean() if "Std" in df.columns else None
                    summary["File"] = fpath
                    # 提取标签（HDP_single_cure 等）
                    label = os.path.basename(subdir)
                    summary["Experiment"] = label
                    rows.append(summary)
    if not rows:
        print("⚠️ No summary files found.")
        return None
    df_all = pd.DataFrame(rows)
    return df_all

# ===========================================================
# 主流程
# ===========================================================
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"🌍 Starting full experiment suite at {timestamp}")
    print(f"Results will be saved under {RESULT_ROOT}\n")

    # 四组实验配置
    experiments = [
        # === HDP (异质) ===
        {"env": "HDP", "mode": "single", "src_dir": f"{DATA_ROOT}/AEEEM", "src_file": "EQ.arff", "tgt_file": f"{DATA_ROOT}/NASA/CM1.arff"},
        {"env": "HDP", "mode": "multi",  "src_dir": f"{DATA_ROOT}/AEEEM", "src_file": None,     "tgt_file": f"{DATA_ROOT}/NASA/CM1.arff"},
        # === IDP (同质) ===
        {"env": "IDP", "mode": "single", "src_dir": f"{DATA_ROOT}/NASA",  "src_file": "MW1.arff", "tgt_file": f"{DATA_ROOT}/NASA/CM1.arff"},
        {"env": "IDP", "mode": "multi",  "src_dir": f"{DATA_ROOT}/NASA",  "src_file": None,      "tgt_file": f"{DATA_ROOT}/NASA/CM1.arff"},
    ]

    # 循环运行实验
    for exp in experiments:
        env, mode = exp["env"], exp["mode"]
        src_dir, tgt_file, src_file = exp["src_dir"], exp["tgt_file"], exp["src_file"]

        # ---- Baseline ----
        base_out = os.path.join(RESULT_ROOT, f"{env}_{mode}_baselines")
        os.makedirs(base_out, exist_ok=True)
        base_args = {
            "src_dirs": src_dir,
            "tgt_file": tgt_file,
            "methods": "none,smote,adasyn,tca,coral,RandomOverSampler,BorderlineSMOTE,SVMSMOTE,KMeansSMOTE,SMOTEENN,SMOTETomek",
            "save_dir": base_out,
            "repeats": REPEATS,
            "cpu": True
        }
        run_experiment("run_baselines.py", base_args, f"{env}-{mode}-BASELINE")

        # ---- CURE ----
        cure_out = os.path.join(RESULT_ROOT, f"{env}_{mode}_cure")
        os.makedirs(cure_out, exist_ok=True)
        cure_args = {
            "arff_src_dirs": src_dir,
            "arff_tgt_file": tgt_file,
            "save_dir": cure_out,
            "repeats": REPEATS,
            "cpu": True
        }
        if src_file:
            cure_args["src_file"] = src_file

        run_experiment("run_cure.py", cure_args, f"{env}-{mode}-CURE")

    # =======================================================
    # 统一汇总所有 summary 文件
    # =======================================================
    print("\n📊 Collecting all summary results...")
    df_all = collect_all_results(RESULT_ROOT)
    if df_all is not None:
        summary_path = os.path.join(RESULT_ROOT, "summary_all.csv")
        df_all.to_csv(summary_path, index=False)
        print(f"✅ All summaries merged to: {summary_path}")
    else:
        print("⚠️ No valid summary files found for merging.")

    print("\n🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"📁 All results saved under: {RESULT_ROOT}")


if __name__ == "__main__":
    main()