#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_hdp_experiments.py (final integrated version)
Supports base models (FMT, CLSUP, MSMDA, CPDP_IFS) and CURE-enhanced variants.
Computes ACC, Pd, Pf, Precision, Recall, F1, Balance, MCC, GMean, AUC.
Includes 95% CI and significance testing (p-values).
"""

import os
import argparse
import numpy as np
import pandas as pd
import subprocess
from tqdm import tqdm
from scipy import stats
from utils_data import load_arff
from hdp_models import HDPModel
from sklearn.metrics import (
    precision_recall_fscore_support,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)

# ==========================================================
# 🔧 Metric calculation
# ==========================================================
def evaluate_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pd_ = tp / (tp + fn) if (tp + fn) > 0 else 0
    pf_ = fp / (fp + tn) if (fp + tn) > 0 else 0
    balance = 1 - np.sqrt(((0 - pf_)**2 + (1 - pd_)**2) / 2)
    gmean = np.sqrt(pd_ * (1 - pf_))
    return dict(ACC=acc, Precision=precision, Recall=recall, F1=f1,
                Pd=pd_, Pf=pf_, Balance=balance, MCC=mcc, GMean=gmean, AUC=auc)

# ==========================================================
# 🧠 Model runner (baseline)
# ==========================================================
def run_base_model(model_name, Xs, Ys, Xt, Yt, repeats=30):
    metrics_list = []
    for r in range(repeats):
        model = HDPModel(model_name)
        y_prob = model.fit_predict(Xs, Ys, Xt)
        metrics = evaluate_metrics(Yt, y_prob)
        metrics_list.append(metrics)
    df = pd.DataFrame(metrics_list)
    mean = df.mean().to_dict()
    std = df.std(ddof=1).to_dict()
    ci = {
        k: stats.t.interval(0.95, len(df)-1, loc=mean[k], scale=std[k]/np.sqrt(len(df)))
        for k in mean
    }
    return mean, std, ci, df

# ==========================================================
# ✨ Run +CURE enhancement via subprocess
# ==========================================================
def run_cure(source_dir, source_file, target_file, save_dir):
    cure_path = os.path.join(os.getcwd(), "CURE.py")
    os.makedirs(save_dir, exist_ok=True)
    result_csv = os.path.join(save_dir, "results.csv")

    cmd = [
        "python", cure_path,
        "--arff_src_dirs", source_dir,
        "--src_file", source_file,
        "--arff_tgt_file", target_file,
        "--save_dir", save_dir,
        "--cpu"
    ]

    subprocess.run(cmd, check=True)

    if os.path.exists(result_csv):
        df = pd.read_csv(result_csv)
        metrics = dict(zip(df["Metric"], df["Value"]))
        for k in metrics:
            metrics[k] = float(metrics[k])
        metrics["ACC"] = (metrics["Pd"] + (1 - metrics["Pf"])) / 2  # approximate ACC
        return metrics
    else:
        raise FileNotFoundError(f"CURE results not found at {result_csv}")

# ==========================================================
# 🚀 Main experiment driver
# ==========================================================
def main():
    parser = argparse.ArgumentParser(description="Cross-project defect prediction with HDP + CURE integration")
    parser.add_argument("--data_root", required=True, help="Path to dataset root")
    parser.add_argument("--save_dir", default="./results_P2P_HDP_CURE", help="Save directory")
    parser.add_argument("--models", default="FMT,CLSUP,MSMDA,CPDP_IFS", help="Comma-separated model names")
    parser.add_argument("--repeats", type=int, default=30, help="Repetitions per experiment")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    base_models = [m.strip() for m in args.models.split(",") if m.strip()]
    datasets = ["AEEEM", "NASA", "JIRA", "PROMISE"]

    # Collect all ARFF files
    all_projects = {
        ds: [
            os.path.join(args.data_root, ds, f)
            for f in os.listdir(os.path.join(args.data_root, ds))
            if f.endswith(".arff")
        ]
        for ds in datasets
        if os.path.isdir(os.path.join(args.data_root, ds))
    }

    all_summary = []
    stats_tests = []

    # Iterate over targets
    for target_ds in datasets:
        if target_ds not in all_projects:
            continue
        for target_file in tqdm(all_projects[target_ds], desc=f"Target={target_ds}"):
            target_name = os.path.basename(target_file)
            Xt, Yt = load_arff(target_file)

            # Iterate over sources
            for source_ds in datasets:
                if source_ds == target_ds:
                    continue
                for source_file in all_projects[source_ds]:
                    source_name = os.path.basename(source_file)
                    print(f"\n🧭 {source_ds}/{source_name} → {target_ds}/{target_name}")
                    Xs, Ys = load_arff(source_file)

                    try:
                        for model_name in base_models:
                            print(f"🔬 Running {model_name}...")
                            mean, std, ci, df = run_base_model(model_name, Xs, Ys, Xt, Yt, repeats=args.repeats)

                            row = {"Source": source_name, "Target": target_name, "Model": model_name}
                            for m in mean:
                                row[f"{m}_mean"] = mean[m]
                                row[f"{m}_std"] = std[m]
                                row[f"{m}_ci_lower"], row[f"{m}_ci_upper"] = ci[m]
                            all_summary.append(row)

                            out_path = os.path.join(args.save_dir, f"{model_name}_results.csv")
                            pd.DataFrame([row]).to_csv(out_path, mode="a", index=False, header=not os.path.exists(out_path))

                        # --- Run +CURE enhancement
                        for model_name in base_models:
                            print(f"✨ Running {model_name}+CURE...")
                            cure_metrics = run_cure(os.path.join(args.data_root, source_ds),
                                                    source_name,
                                                    target_file,
                                                    os.path.join(args.save_dir, f"{model_name}_CURE"))
                            row = {"Source": source_name, "Target": target_name, "Model": model_name + "+CURE"}
                            for k, v in cure_metrics.items():
                                row[k + "_mean"] = v
                            all_summary.append(row)

                            out_path = os.path.join(args.save_dir, f"{model_name}_CURE_results.csv")
                            pd.DataFrame([row]).to_csv(out_path, mode="a", index=False, header=not os.path.exists(out_path))

                        # --- Significance tests (base vs +CURE)
                        for model_name in base_models:
                            base_df = pd.read_csv(os.path.join(args.save_dir, f"{model_name}_results.csv"))
                            cure_df = pd.read_csv(os.path.join(args.save_dir, f"{model_name}_CURE_results.csv"))
                            if len(base_df) > 0 and len(cure_df) > 0:
                                base_acc = base_df["ACC_mean"].dropna().values
                                cure_acc = cure_df["ACC_mean"].dropna().values
                                if len(base_acc) > 1 and len(cure_acc) > 1:
                                    _, p = stats.ttest_ind(base_acc, cure_acc, equal_var=False)
                                    stats_tests.append({
                                        "Source": source_name,
                                        "Target": target_name,
                                        "Compare": f"{model_name} vs {model_name}+CURE",
                                        "p_value": p
                                    })

                    except Exception as e:
                        print(f"❌ Error {source_name}→{target_name}: {e}")

    # Save final summary and significance results
    summary_path = os.path.join(args.save_dir, "results_summary.csv")
    pd.DataFrame(all_summary).to_csv(summary_path, index=False)
    print(f"\n✅ All results saved to {summary_path}")

    if stats_tests:
        pd.DataFrame(stats_tests).to_csv(os.path.join(args.save_dir, "stats_tests.csv"), index=False)
        print("📊 Significance tests saved (p-values included).")


if __name__ == "__main__":
    main()