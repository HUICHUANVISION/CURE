#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run modern deep HDP baselines (DANN, ADDA, Deep CORAL) on all transfer pairs."""

from __future__ import annotations

import argparse, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score
from tqdm import tqdm

from utils_data import load_arff
from deep_hdp_baselines import fit_predict_adda, fit_predict_dann, fit_predict_deep_coral

METHODS = {
    "DANN": fit_predict_dann,
    "ADDA": fit_predict_adda,
    "Deep CORAL": fit_predict_deep_coral,
}
BASE_MODELS = ["CPDP_IFS", "CLSUP", "FMT", "MSMDA"]  # labels retained for comparison strata
METRICS = ["AUC", "F1", "GMean", "MCC", "Pd", "Pf"]


def evaluate_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pd_ = tp / (tp + fn) if (tp + fn) else 0.0
    pf_ = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "AUC": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5,
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "GMean": float(np.sqrt(max(0.0, pd_ * (1.0 - pf_)))),
        "MCC": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "Pd": float(pd_),
        "Pf": float(pf_),
    }


def discover_projects(data_root, datasets):
    return {
        ds: sorted(str(p) for p in (Path(data_root) / ds).glob("*.arff"))
        for ds in datasets if (Path(data_root) / ds).is_dir()
    }


def load_existing(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("records", [])
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/Users/charles/Documents/moss_paper/CURE/CURE_code/Datasets")
    ap.add_argument("--output", default="/Users/charles/Documents/moss_paper/CURE/deep_baseline_results.json")
    ap.add_argument("--datasets", default="AEEEM,NASA,JIRA,PROMISE")
    ap.add_argument("--methods", default="DANN,ADDA,Deep CORAL")
    ap.add_argument("--base_models", default=",".join(BASE_MODELS))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    base_models = [x.strip() for x in args.base_models.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    projects = discover_projects(args.data_root, datasets)

    records = [] if args.no_resume else load_existing(args.output)
    done = {(r["SourceDataset"], r["Source"], r["TargetDataset"], r["Target"], r["Method"], r["BaseModel"], r["Seed"]) for r in records}
    total_pairs = sum(len(projects.get(s, [])) * len(projects.get(t, [])) for s in datasets for t in datasets if s != t)
    print(f"Discovered {total_pairs} transfer pairs; existing records={len(records)}")

    count = 0
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for target_ds in datasets:
        for target_file in tqdm(projects.get(target_ds, []), desc=f"Target={target_ds}"):
            Xt, Yt = load_arff(target_file)
            for source_ds in datasets:
                if source_ds == target_ds:
                    continue
                for source_file in projects.get(source_ds, []):
                    Xs, Ys = load_arff(source_file)
                    for method in methods:
                        fn = METHODS[method]
                        for seed in seeds:
                            missing_base = []
                            for base_model in base_models:
                                key = (source_ds, os.path.basename(source_file), target_ds, os.path.basename(target_file), method, base_model, seed)
                                if key not in done:
                                    missing_base.append(base_model)
                            if not missing_base:
                                continue
                            try:
                                y_prob = fn(Xs, Ys, Xt, seed=seed, epochs=args.epochs, latent_dim=args.latent_dim, lr=args.lr)
                                metrics = evaluate_metrics(Yt, y_prob)
                                status, err = "ok", ""
                            except Exception as e:
                                metrics = {m: None for m in METRICS}
                                status, err = "error", repr(e)
                            for base_model in missing_base:
                                key = (source_ds, os.path.basename(source_file), target_ds, os.path.basename(target_file), method, base_model, seed)
                                rec = {
                                    "SourceDataset": source_ds, "Source": os.path.basename(source_file),
                                    "TargetDataset": target_ds, "Target": os.path.basename(target_file),
                                    "BaseModel": base_model, "Method": method, "Seed": seed,
                                    "epochs": args.epochs, "latent_dim": args.latent_dim, "lr": args.lr,
                                    "status": status, "error": err, **metrics,
                                }
                                records.append(rec); done.add(key)
                            count += len(missing_base)
                            if count % 24 == 0:
                                with open(out, "w", encoding="utf-8") as f: json.dump(records, f, indent=2)
                            if args.limit and count >= args.limit:
                                with open(out, "w", encoding="utf-8") as f: json.dump(records, f, indent=2)
                                print(f"Stopped at limit={args.limit}")
                                return
    with open(out, "w", encoding="utf-8") as f: json.dump(records, f, indent=2)
    df = pd.DataFrame(records)
    csv_path = out.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(records)} records to {out} and {csv_path} in {(time.time()-t0)/60:.1f} min")
    if not df.empty:
        print(df[df.status == "ok"].groupby("Method")[METRICS].mean(numeric_only=True).round(4))


if __name__ == "__main__":
    main()
