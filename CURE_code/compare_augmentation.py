#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版 compare_augmentation.py：
- 取消 timeout 限制（除非用户显式提供）
- 自动适配 Mac M1/M2 (MPS) 芯片提示
- 降低数据生成失败敏感度（tau 更低、权重更松）
- 避免生成失败时中断整个流程
"""

import os
import sys
import argparse
import subprocess
import shlex
import re
import json
import math
import pandas as pd
from datetime import datetime
from pathlib import Path

PROGRESS_PAT = re.compile(r"(\[(Align|Gen|Final)[^\]]*\]|Validation:|Test:|\u5408\u6210\u6837\u672c|Synthesis|\u6e90\u6743\u91cd rho)")
VAL_RE  = re.compile(r"^Validation:\s*(\{.*\})\s*$")
TEST_RE = re.compile(r"^Test:\s*(\{.*\})\s*$")

VAL_PDPFGM_RE  = re.compile(r"^Val\s+Pd\s*=\s*([0-9.]+)\s*,\s*Pf\s*=\s*([0-9.]+)\s*,\s*GM\s*=\s*([0-9.]+)\s*$", re.I)
TEST_PDPFGM_RE = re.compile(r"^Test\s+Pd\s*=\s*([0-9.]+)\s*,\s*Pf\s*=\s*([0-9.]+)\s*,\s*GM\s*=\s*([0-9.]+)\s*$", re.I)

VAL_CONF_RE  = re.compile(r"^Val\s+Confusion:\s*TP\s*=\s*(\d+)\s*,\s*FP\s*=\s*(\d+)\s*,\s*TN\s*=\s*(\d+)\s*,\s*FN\s*=\s*(\d+)\s*$", re.I)
TEST_CONF_RE = re.compile(r"^Test\s+Confusion:\s*TP\s*=\s*(\d+)\s*,\s*FP\s*=\s*(\d+)\s*,\s*TN\s*=\s*(\d+)\s*,\s*FN\s*=\s*(\d+)\s*$", re.I)

def try_json(s):
    try:
        return json.loads(s.replace("'", '"'))
    except Exception:
        return {}

def parse_metrics_and_pdpfgm_from_text(text):
    metrics_val, metrics_test = {}, {}
    val_pd = val_pf = val_gm = test_pd = test_pf = test_gm = None

    for line in text.splitlines():
        s = line.strip()
        if (m := VAL_RE.match(s)): metrics_val = try_json(m.group(1))
        if (m := TEST_RE.match(s)): metrics_test = try_json(m.group(1))
        if (m := VAL_PDPFGM_RE.match(s)): val_pd, val_pf, val_gm = map(float, m.groups())
        if (m := TEST_PDPFGM_RE.match(s)): test_pd, test_pf, test_gm = map(float, m.groups())

    for line in text.splitlines():
        s = line.strip()
        if None in (val_pd, val_pf, val_gm):
            if (m := VAL_CONF_RE.match(s)):
                TP, FP, TN, FN = map(int, m.groups())
                val_pd = TP / (TP + FN) if (TP + FN) > 0 else None
                val_pf = FP / (FP + TN) if (FP + TN) > 0 else None
                spc = 1 - val_pf if val_pf is not None else None
                val_gm = math.sqrt(val_pd * spc) if (val_pd is not None and spc is not None) else None
        if None in (test_pd, test_pf, test_gm):
            if (m := TEST_CONF_RE.match(s)):
                TP, FP, TN, FN = map(int, m.groups())
                test_pd = TP / (TP + FN) if (TP + FN) > 0 else None
                test_pf = FP / (FP + TN) if (FP + TN) > 0 else None
                spc = 1 - test_pf if test_pf is not None else None
                test_gm = math.sqrt(test_pd * spc) if (test_pd is not None and spc is not None) else None

    return metrics_val, metrics_test, {
        "Val_Pd": val_pd, "Val_Pf": val_pf, "Val_GM": val_gm,
        "Test_Pd": test_pd, "Test_Pf": test_pf, "Test_GM": test_gm,
    }

def build_cmd(python_exec, script_path, common_flags, extra_flags, outdir, mode, seed, cpu, val_threshold_mode, quick):
    flags = common_flags.strip()
    if quick and "--align_epochs" not in flags: flags += " --align_epochs 3"
    if quick and "--gen_epochs" not in flags: flags += " --gen_epochs 5"
    if quick and "--final_epochs" not in flags: flags += " --final_epochs 5"
    if val_threshold_mode and f"--val_threshold_mode {val_threshold_mode}" not in flags:
        flags += f" --val_threshold_mode {val_threshold_mode}"
    if cpu and "--cpu" not in flags: flags += " --cpu"
    save_dir = f"{outdir}/{mode}_s{seed}"
    flags += f" --save_dir {save_dir}"
    flags += " " + " ".join(shlex.quote(x) for x in extra_flags)
    return f"{shlex.quote(python_exec)} {shlex.quote(str(script_path))} {flags}", save_dir

def run_once(mode, args, extra_flags, seed):
    cmd, save_dir = build_cmd(args.python, args.script, args.common_flags, extra_flags, args.outdir, mode, seed, args.cpu, args.val_threshold_mode, args.quick)
    print(f"\n[RUN][{mode}][seed={seed}] {cmd}")
    stdout_file = Path(args.outdir) / f"{mode}_s{seed}.stdout.txt"
    stderr_file = Path(args.outdir) / f"{mode}_s{seed}.stderr.txt"

    with open(stdout_file, "w", encoding="utf-8") as fout, open(stderr_file, "w", encoding="utf-8") as ferr:
        proc = subprocess.Popen(cmd, shell=True, cwd=args.workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        captured_text = []
        import time, select
        while True:
            rlist, _, _ = select.select([proc.stdout, proc.stderr], [], [], 0.25)
            for stream in rlist:
                line = stream.readline()
                if not line: continue
                if stream is proc.stdout: fout.write(line); fout.flush()
                else: ferr.write(line); ferr.flush()
                captured_text.append(line)
                if PROGRESS_PAT.search(line): print(line.rstrip())
            if proc.poll() is not None:
                rest_out = proc.stdout.read(); rest_err = proc.stderr.read()
                if rest_out: fout.write(rest_out); captured_text.append(rest_out)
                if rest_err: ferr.write(rest_err)
                break
        try: proc.kill()
        except: pass

    return parse_metrics_and_pdpfgm_from_text("".join(captured_text))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--script", required=True)
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--outdir", default="./runs/compare_aug")
    ap.add_argument("--common_flags", default="")
    ap.add_argument("--val_threshold_mode", default="", choices=["", "f1", "youden"])
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin" and not args.cpu:
        print("[提示] 当前为 macOS，建议使用 --cpu 或手动检查 MPS 支持。")

    all_rows = []
    for seed in args.seeds:
        print(f"=== Seed {seed} ===")
        base_flags = ["--n_gen_per_seed", "0", "--lambda_syn", "0", "--mu_mmd", "0", "--seed", str(seed)]
        aug_flags = ["--n_gen_per_seed", "8", "--tau", "0.5", "--synth_min_weight", "1e-5", "--lambda_syn", "1.5", "--mu_mmd", "0.03", "--seed", str(seed)]
        if args.cpu: base_flags.append("--cpu"); aug_flags.append("--cpu")

        val_b, test_b, extra_b = run_once("baseline", args, base_flags, seed)
        val_a, test_a, extra_a = run_once("augmented", args, aug_flags, seed)

        row = {"Seed": seed}
        for k, v in (val_b or {}).items(): row[f"Val_{k}_Base"] = v
        for k, v in (test_b or {}).items(): row[f"Test_{k}_Base"] = v
        for k, v in (val_a or {}).items(): row[f"Val_{k}_Aug"] = v
        for k, v in (test_a or {}).items(): row[f"Test_{k}_Aug"] = v
        for k, v in (extra_b or {}).items(): row[f"{k}_Base"] = v
        for k, v in (extra_a or {}).items(): row[f"{k}_Aug"] = v
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(args.outdir) / f"compare_{ts}.csv"
    md_path = Path(args.outdir) / f"compare_{ts}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")

    print(f"\n[OK] 对比完成：\n- CSV: {csv_path}\n- MD:  {md_path}\n日志：{args.outdir}/*stdout.txt / *stderr.txt")

if __name__ == "__main__":
    main()

