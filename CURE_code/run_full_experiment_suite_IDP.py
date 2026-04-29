#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_full_experiment_suite.py
🌍 全自动 IDP（同质环境）实验脚本
每个 NASA 数据集轮流作为 target：
- 单源：其他每个单独的 source
- 多源：所有其他项目联合为 source
"""

import os
import subprocess
from datetime import datetime

# =========================
# 配置
# =========================
DATASET_DIR = "//Datasets/NASA"
SAVE_ROOT = "./results_suite"
REPEATS = 30
METHODS = "none,smote,adasyn,tca,coral,RandomOverSampler,BorderlineSMOTE,SVMSMOTE,KMeansSMOTE,SMOTEENN,SMOTETomek"

# =========================
# 工具函数
# =========================
def run_experiment(script, args, tag):
    """运行单个实验任务"""
    print(f"\n🚀 Running [{tag}] ...")
    cmd = ["python", script] + args
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

def list_arff_files(folder):
    """列出目录中所有 .arff 文件"""
    return sorted([f for f in os.listdir(folder) if f.endswith(".arff")])

# =========================
# 主流程
# =========================
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"🌍 Starting full IDP experiment suite at {timestamp}")
    print(f"Results will be saved under {SAVE_ROOT}\n")

    os.makedirs(SAVE_ROOT, exist_ok=True)
    nasa_files = list_arff_files(DATASET_DIR)
    print(f"🧩 Found {len(nasa_files)} NASA datasets: {nasa_files}")

    # ========== 遍历每个 target ==========
    for tgt_file in nasa_files:
        tgt_path = os.path.join(DATASET_DIR, tgt_file)
        tgt_name = os.path.splitext(tgt_file)[0]

        # ========== 单源实验 ==========
        for src_file in nasa_files:
            if src_file == tgt_file:
                continue

            src_name = os.path.splitext(src_file)[0]

            # CURE 单源
            cure_single_dir = os.path.join(SAVE_ROOT, f"IDP_single_CURE_{src_name}to{tgt_name}")
            args_cure_single = [
                "--arff_src_dirs", DATASET_DIR,
                "--src_file", src_file,
                "--arff_tgt_file", tgt_path,
                "--save_dir", cure_single_dir,
                "--repeats", str(REPEATS),
                "--cpu"
            ]
            run_experiment("run_cure.py", args_cure_single, f"IDP-single-CURE {src_name}→{tgt_name}")

            # Baseline 单源
            base_single_dir = os.path.join(SAVE_ROOT, f"IDP_single_BASE_{src_name}to{tgt_name}")
            args_base_single = [
                "--src_dirs", DATASET_DIR,
                "--src_file", src_file,
                "--tgt_file", tgt_path,
                "--methods", METHODS,
                "--save_dir", base_single_dir,
                "--repeats", str(REPEATS),
                "--cpu"
            ]
            run_experiment("run_baselines.py", args_base_single, f"IDP-single-BASELINE {src_name}→{tgt_name}")

        # ========== 多源实验 ==========
        cure_multi_dir = os.path.join(SAVE_ROOT, f"IDP_multi_CURE_to{tgt_name}")
        args_cure_multi = [
            "--arff_src_dirs", DATASET_DIR,
            "--arff_tgt_file", tgt_path,
            "--save_dir", cure_multi_dir,
            "--repeats", str(REPEATS),
            "--cpu"
        ]
        run_experiment("run_cure.py", args_cure_multi, f"IDP-multi-CURE →{tgt_name}")

        base_multi_dir = os.path.join(SAVE_ROOT, f"IDP_multi_BASE_to{tgt_name}")
        args_base_multi = [
            "--src_dirs", DATASET_DIR,
            "--tgt_file", tgt_path,
            "--methods", METHODS,
            "--save_dir", base_multi_dir,
            "--repeats", str(REPEATS),
            "--cpu"
        ]
        run_experiment("run_baselines.py", args_base_multi, f"IDP-multi-BASELINE →{tgt_name}")

    print("\n✅ All IDP experiments finished successfully!")


if __name__ == "__main__":
    main()