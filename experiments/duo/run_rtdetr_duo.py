"""RT-DETR-R50 on DUO using the existing 416 no-zoom recipe."""

import argparse
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path("/root/duo_exp")
PROJECT = Path("/root/autodl-tmp/RT-DETR/rtdetr_pytorch")
PYTHON = "/root/miniconda3/bin/python"
BASE = PROJECT / "configs/rtdetr/rtdetr_r50vd_urpc_416_nozoom.yml"


def configs(run: Path):
    data_dir = PROJECT / "configs/dataset"
    source = yaml.safe_load((data_dir / "urpc_detection.yml").read_text())
    for split in ("train", "test"):
        data = json.loads(json.dumps(source))
        tr = data["train_dataloader"]["dataset"]
        tr["img_folder"] = str(ROOT / "data/raw/DUO/images/train")
        tr["ann_file"] = str(ROOT / "data/annotations/instances_train.json")
        val = data["val_dataloader"]["dataset"]
        val["img_folder"] = str(ROOT / f"data/raw/DUO/images/{'val' if split == 'train' else 'test'}")
        val["ann_file"] = str(ROOT / f"data/annotations/instances_{'val' if split == 'train' else 'test'}.json")
        (data_dir / f"duo_{split}_detection.yml").write_text(yaml.safe_dump(data, sort_keys=False))
        text = BASE.read_text().replace("../dataset/urpc_detection.yml", f"../dataset/duo_{split}_detection.yml")
        text = text.replace("/root/autodl-tmp/work_dirs/rtdetr_r50vd_urpc_416_nozoom", str(run))
        (PROJECT / f"configs/rtdetr/rtdetr_r50vd_duo_{split}.yml").write_text(text)
    return PROJECT / "configs/rtdetr/rtdetr_r50vd_duo_train.yml", PROJECT / "configs/rtdetr/rtdetr_r50vd_duo_test.yml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    run = ROOT / "runs" / f"rtdetr_seed{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    train_cfg, test_cfg = configs(run)
    command = [PYTHON, "tools/train.py", "-c", str(train_cfg), "-t",
               "/root/autodl-tmp/RT-DETR/weights/rtdetr_r50vd_6x_coco.pth",
               "--amp", "--seed", str(args.seed)]
    with (run / "train.log").open("w") as log:
        subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
    checkpoint = run / "checkpoint.pth"
    assert checkpoint.exists(), checkpoint
    subprocess.run([PYTHON, "-u", str(ROOT / "eval_rtdetr_duo.py"),
                    "--config", str(test_cfg), "--ckpt", str(checkpoint),
                    "--ann", str(ROOT / "data/annotations/instances_test.json"),
                    "--out", str(run / "test_eval.json")], cwd=PROJECT, check=True)
    for path in run.glob("checkpoint[0-9]*.pth"):
        path.unlink()


if __name__ == "__main__":
    main()
