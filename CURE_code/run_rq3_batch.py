#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_rq3_batch.py
Batch runner for RQ3 ablation study of CURE.
Runs all dataset × (DA, CS, WS) × seeds combinations automatically.
"""

import os
import itertools
import subprocess

# 配置部分 ==========================
datasets = {
    # "AEEEM": ["EQ", "JDT", "Lucene", "Mylyn", "PDE"],
    "JIRA":  ["activemq-5.0.0", "derby-10.5.1.1", "groovy-1_6_BETA_1", "hbase-0.94.0", "hive-0.9.0", "jruby-1.1", "wicket-1.3.0-beta2"],
    # "NASA":  ["CM1", "PC1", "PC2", "MW1", "PC3"],
    "PROMISE": ["ant-1.7", "camel-1.4", "ivy-2.0", "jedit-4.0", "log4j-1.0", "poi-2.0", "tomcat", "velocity-1.6", "xalan-2.4", "xerces-1.3"]
}

da_options = ["none", "mmd"]
cs_options = [False, True]
ws_options = [False, True]
seeds = [1, 2, 3, 4, 5]

base_dir = "./Datasets"      # 数据集根目录
save_root = "./runs/RQ3_all" # 输出根目录
n_gen = 200                  # 每类合成样本数
# ===================================

os.makedirs(save_root, exist_ok=True)

def run_one(dataset_name, target, da, cs, ws, seed):
    """构造命令行并执行"""
    tgt_path = os.path.join(base_dir, dataset_name, f"{target}.arff")
    src_dir = os.path.join(base_dir, dataset_name)
    save_dir = os.path.join(save_root, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        "python", "CURE_RQ3_ablation.py",
        "--arff_src_dirs", src_dir,
        "--arff_tgt_file", tgt_path,
        "--da", da,
        "--n_gen_per_seed", str(n_gen),
        "--save_dir", save_dir,
        "--seed", str(seed)
    ]
    if cs: cmd.append("--cs")
    if ws: cmd.append("--ws")

    print("🚀", " ".join(cmd))
    subprocess.run(cmd)

if __name__ == "__main__":
    for dataset_name, targets in datasets.items():
        for target, da, cs, ws, seed in itertools.product(targets, da_options, cs_options, ws_options, seeds):
            run_one(dataset_name, target, da, cs, ws, seed)