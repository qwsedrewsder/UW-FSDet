"""D-FINE-S on DUO using the existing 416-pixel URPC configuration."""

import argparse
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path("/root/duo_exp")
PROJECT = Path("/root/autodl-tmp/D-FINE")
PYTHON = "/root/miniconda3/bin/python"
BASE = PROJECT / "configs/dfine/custom/dfine_hgnetv2_s_urpc.yml"


def configs(run: Path):
    dataset_dir = PROJECT / "configs/dataset"
    base_data = yaml.safe_load((dataset_dir / "urpc_detection.yml").read_text())
    for split, val_split in (("train", "val"), ("test", "test")):
        data = json.loads(json.dumps(base_data))
        val = data["val_dataloader"]["dataset"]
        val["img_folder"] = str(ROOT / f"data/raw/DUO/images/{val_split}")
        val["ann_file"] = str(ROOT / f"data/annotations/instances_{val_split}.json")
        if split == "train":
            tr = data["train_dataloader"]["dataset"]
            tr["img_folder"] = str(ROOT / "data/raw/DUO/images/train")
            tr["ann_file"] = str(ROOT / "data/annotations/instances_train.json")
        target_data = dataset_dir / f"duo_{split}_detection.yml"
        target_data.write_text(yaml.safe_dump(data, sort_keys=False))
        text = BASE.read_text().replace("../../dataset/urpc_detection.yml",
                                        f"../../dataset/duo_{split}_detection.yml")
        text = text.replace("/root/autodl-tmp/work_dirs/dfine_s_urpc", str(run))
        target_cfg = PROJECT / f"configs/dfine/custom/dfine_s_duo_{split}.yml"
        target_cfg.write_text(text)
    return PROJECT / "configs/dfine/custom/dfine_s_duo_train.yml", PROJECT / "configs/dfine/custom/dfine_s_duo_test.yml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    run = ROOT / "runs" / f"dfine_seed{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    train_cfg, test_cfg = configs(run)
    subprocess.run([PYTHON, "train.py", "-c", str(train_cfg), "-d", "cuda:0",
                    "--seed", str(args.seed), "--use-amp", "--output-dir", str(run)],
                   cwd=PROJECT, check=True)
    checkpoint = run / "last.pth"
    if not checkpoint.exists():
        checkpoint = run / "checkpoint.pth"
    assert checkpoint.exists(), checkpoint
    pred = run / "test_predictions.json"
    subprocess.run([PYTHON, "-u", str(ROOT / "eval_dfine_capture.py"), "-c",
                    str(test_cfg), "-r", str(checkpoint), "--pred-out", str(pred),
                    "--seed", str(args.seed)], cwd=PROJECT, check=True)
    subprocess.run([PYTHON, "-u", str(ROOT / "eval_dfine_metrics.py"), "--pred",
                    str(pred), "--out", str(run / "test_eval.json"), "--ann",
                    str(ROOT / "data/annotations/instances_test.json")], check=True)
    pred.unlink(missing_ok=True)
    for path in run.glob("checkpoint[0-9]*.pth"):
        path.unlink()


if __name__ == "__main__":
    main()
