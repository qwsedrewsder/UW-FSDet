"""DINO-4scale on DUO with its established 416-pixel training recipe."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/root/duo_exp")
PROJECT = "/root/autodl-tmp/DINO"
PYTHON = "/root/miniconda3/bin/python"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    run = ROOT / "runs" / f"dino_seed{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    command = [PYTHON, "-u", "main.py", "-c", "config/DINO/urpc_4scale_416.py",
               "--coco_path", str(ROOT / "data/dino_train"), "--output_dir", str(run),
               "--pretrain_model_path", f"{PROJECT}/weights/checkpoint0011_4scale.pth",
               "--finetune_ignore", "label_enc.weight", "class_embed", "--options",
               "dn_scalar=100", "embed_init_tgt=TRUE", "dn_label_coef=1.0",
               "dn_bbox_coef=1.0", "use_ema=False", "dn_box_noise_scale=1.0",
               "batch_size=16", "--seed", str(args.seed), "--num_workers", "8"]
    with (run / "train.log").open("w") as log:
        subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
    checkpoint = run / "checkpoint.pth"
    assert checkpoint.exists(), checkpoint
    subprocess.run([PYTHON, "-u", str(ROOT / "eval_dino_duo.py"), "--seed", str(args.seed)],
                   cwd=PROJECT, check=True)
    for path in run.glob("checkpoint*.pth"):
        if path.name not in {"checkpoint.pth", "checkpoint_best_regular.pth"}:
            path.unlink()


if __name__ == "__main__":
    main()
