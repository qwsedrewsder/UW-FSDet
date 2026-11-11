"""Sequential experiment runner for S3-S6.
python tools/run_matrix.py --config <base.yaml> --matrix <matrix.yaml> [--skip-eval] [--start N]

matrix.yaml: list of run dicts. Each run:
  exp: name (required)
  seed: int (default 0)
  config: override base config (optional)
  skip_train: bool (evaluation-only, e.g. TTA runs)
  epochs: int (override schedule.epochs)
  <dotted.key>: value   -- any extra key becomes a config override
                          (e.g. fg_fpn.fuse_conv: pairwise)
Runs serially; appends each result to work_dirs/EXPERIMENTS.csv.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import yaml

BASE = "/root/autodl-tmp"


def get_config_hash(work_dir):
    import hashlib
    p = os.path.join(work_dir, "config.yaml")
    if os.path.exists(p):
        with open(p, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:12]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True, help="yaml list of run dicts")
    ap.add_argument("--config", default=None, help="base config")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    with open(args.matrix) as f:
        runs = yaml.safe_load(f)

    csv_path = os.path.join(BASE, "work_dirs", "EXPERIMENTS.csv")
    header = ["exp", "seed", "AP", "AP50", "AP75", "APs", "APm", "APl", "params", "time_s", "note", "config_hash"]
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    for i, ov in enumerate(runs):
        if i < args.start:
            continue
        exp_name = ov.get("exp", f"run_{i}")
        seed = ov.get("seed", 0)
        cfg_path = ov.get("config", args.config)
        if cfg_path is None:
            print(f"[matrix {i}] run {i} has no config; skipping", flush=True)
            continue

        # build overrides: dotted keys (except reserved fields)
        reserved = {"exp", "seed", "config", "skip_train", "epochs", "tta", "tta_scales",
                    "tta_flip", "ckpt", "score_thr", "nms_iou", "max_per_img", "soft_nms"}
        set_ov = []
        for k, v in ov.items():
            if k in reserved:
                continue
            set_ov.append(f"{k}={v}")
        if "epochs" in ov:
            set_ov.append(f"schedule.epochs={ov['epochs']}")

        work_dir = os.path.join(BASE, "work_dirs", exp_name)
        os.makedirs(work_dir, exist_ok=True)
        chash = get_config_hash(work_dir)
        t0 = time.time()

        if not ov.get("skip_train", False):
            cmd = [sys.executable, os.path.join(BASE, "htdet", "tools", "train.py"),
                   "--config", cfg_path, "--seed", str(seed), "--work-dir", work_dir]
            for s in set_ov:
                cmd += ["--set", s]
            print(f"[matrix {i}/{len(runs)}] {exp_name} seed={seed} ov={ {k: v for k, v in ov.items() if k not in ('exp',)} }", flush=True)
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"FAILED {exp_name}: rc={r.returncode}", flush=True)
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([exp_name, seed, "TRAIN_FAIL", "", "", "", "", "", "", "", "", chash])
                continue
            chash = get_config_hash(work_dir)
        else:
            print(f"[matrix {i}/{len(runs)}] {exp_name} (eval-only)", flush=True)

        if not args.skip_eval:
            # TTA runs specify tta params
            tta = ov.get("tta", False)
            tta_scales = ov.get("tta_scales", "320,416")
            tta_flip = ov.get("tta_flip", True)
            ckpt = ov.get("ckpt", os.path.join(work_dir, "ckpt", "final.pth"))
            if not os.path.exists(ckpt):
                ckpt_dir = os.path.join(work_dir, "ckpt")
                cands = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]) if os.path.isdir(ckpt_dir) else []
                ckpt = os.path.join(ckpt_dir, cands[-1]) if cands else None
            if ckpt and os.path.exists(ckpt):
                # use the dumped full config in work_dir (train wrote it);
                # for skip_train (TTA) runs cfg_path points at the training run
                eval_cfg = os.path.join(work_dir, "config.yaml")
                if not os.path.exists(eval_cfg):
                    eval_cfg = cfg_path
                cmd = [sys.executable, os.path.join(BASE, "htdet", "tools", "test.py"),
                       "--config", eval_cfg, "--ckpt", ckpt,
                       "--out", os.path.join(work_dir, "eval.json")]
                if "score_thr" in ov:
                    cmd += ["--score-thr", str(ov["score_thr"])]
                if "nms_iou" in ov:
                    cmd += ["--nms-iou", str(ov["nms_iou"])]
                if "max_per_img" in ov:
                    cmd += ["--max-per-img", str(ov["max_per_img"])]
                if ov.get("soft_nms", False):
                    cmd += ["--soft-nms"]
                if tta:
                    cmd += ["--tta", "--tta-scales", str(tta_scales)]
                    if tta_flip:
                        cmd += ["--tta-flip"]
                r = subprocess.run(cmd)
                if r.returncode == 0 and os.path.exists(os.path.join(work_dir, "eval.json")):
                    with open(os.path.join(work_dir, "eval.json")) as f:
                        ev = json.load(f)
                    with open(csv_path, "a", newline="") as f:
                        w = csv.writer(f)
                        w.writerow([exp_name, seed, round(ev["AP"], 4), round(ev["AP50"], 4),
                                    round(ev["AP75"], 4), round(ev["APs"], 4),
                                    round(ev["APm"], 4), round(ev["APl"], 4),
                                    ev["params"], round(time.time() - t0),
                                    f"tta={tta}", chash])
                else:
                    with open(csv_path, "a", newline="") as f:
                        csv.writer(f).writerow([exp_name, seed, "EVAL_FAIL", "", "", "", "", "", "", "", "", chash])
            else:
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([exp_name, seed, "NO_CKPT", "", "", "", "", "", "", "", "", chash])
        else:
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([exp_name, seed, "SKIP_EVAL", "", "", "", "", "", "", "", "", chash])
        print(f"[matrix] {exp_name} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
