"""DECO R50 on DUO with its established 416-pixel recipe."""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path("/root/duo_exp")
PROJECT = Path("/root/autodl-tmp/DECO/deco")
PYTHON = "/root/miniconda3/bin/python"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    run = ROOT / "runs" / f"deco_seed{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    command = [PYTHON, "-u", "main.py", "--model", "deco", "--backbone", "resnet50",
               "--batch_size", "8", "--epochs", "150", "--lr_drop", "100",
               "--coco_path", str(ROOT / "data/dino_train"), "--output_dir", str(run),
               "--device", "cuda", "--seed", str(args.seed),
               "--pretrain_model_path", "weights/deco_r50_150e.pth",
               "--num_classes", "4", "--num_workers", "8"]
    with (run / "train.log").open("w") as log:
        subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
    checkpoint = run / "checkpoint.pth"
    assert checkpoint.exists(), checkpoint
    eval_command = [PYTHON, "-u", str(ROOT / "eval_deco_duo.py"), "--seed",
                    str(args.seed), "--ckpt", str(checkpoint), "--out",
                    str(run / "test_eval.json")]
    subprocess.run(eval_command, cwd=PROJECT, check=True)
    for path in run.glob("checkpoint[0-9]*.pth"):
        path.unlink()


if __name__ == "__main__":
    main()
