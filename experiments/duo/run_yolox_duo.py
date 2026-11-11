"""YOLOX-Tiny on DUO with the existing YOLOX-Tiny 416 recipe."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/root/duo_exp")
PROJECT = Path("/root/autodl-tmp/YOLOX")
PYTHON = "/root/miniconda3/bin/python"


def link_dataset():
    data = ROOT / "data/yolox"
    (data / "annotations").mkdir(parents=True, exist_ok=True)
    for split, alias in (("train", "train2017"), ("val", "val2017"), ("test", "test2017")):
        target = data / alias
        source = ROOT / f"data/yolo/images/{split}"
        if not target.exists():
            target.symlink_to(source, target_is_directory=True)
        ann_source = ROOT / f"data/annotations/instances_{split}.json"
        ann_target = data / "annotations" / f"{split}.json"
        if not ann_target.exists():
            ann_target.symlink_to(ann_source)


def evaluate(checkpoint: Path, seed: int, run: Path):
    import sys
    sys.path.insert(0, str(PROJECT))
    sys.path.insert(0, "/root/autodl-tmp/htdet")
    from yolox.data.data_augment import ValTransform
    from yolox.exp import get_exp
    from yolox.utils import postprocess
    from htdet.engine.evaluator import coco_eval
    exp_path = PROJECT / f"exps/default/yolox_tiny_duo_seed{seed}.py"
    exp = get_exp(str(exp_path), None)
    model = exp.get_model().cuda().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    ann = ROOT / "data/annotations/instances_test.json"
    images = ROOT / "data/yolo/images/test"
    data = json.loads(ann.read_text())
    name_to_id = {im["file_name"]: im["id"] for im in data["images"]}
    files = sorted(name_to_id)
    preproc = ValTransform(legacy=False)
    detections = []
    with torch.no_grad():
        for start in range(0, len(files), 32):
            names, tensors, ratios = [], [], []
            for name in files[start:start + 32]:
                image = cv2.imread(str(images / name))
                h, w = image.shape[:2]
                ratio = min(exp.test_size[0] / h, exp.test_size[1] / w)
                tensor, _ = preproc(image, None, exp.test_size)
                names.append(name); tensors.append(torch.from_numpy(tensor)); ratios.append(ratio)
            output = postprocess(model(torch.stack(tensors).float().cuda()), exp.num_classes,
                                 exp.test_conf, exp.nmsthre, class_agnostic=False)
            for name, result, ratio in zip(names, output, ratios):
                if result is None or len(result) == 0:
                    continue
                boxes = result[:, :4].cpu().numpy().astype(np.float32) / ratio
                scores = (result[:, 4] * result[:, 5]).cpu().numpy()
                labels = result[:, 6].long().cpu().numpy()
                keep = np.argsort(scores)[::-1][:100]
                detections.append((name_to_id[name], torch.tensor(boxes[keep]),
                                   torch.tensor(scores[keep]), torch.tensor(labels[keep])))
    metrics = coco_eval(detections, str(ann))
    (run / "test_eval.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    link_dataset()
    run_name = f"yolox_tiny_seed{args.seed}"
    run = ROOT / "runs" / run_name
    run.mkdir(parents=True, exist_ok=True)
    exp_path = PROJECT / f"exps/default/yolox_tiny_duo_seed{args.seed}.py"
    base = (PROJECT / "exps/default/yolox_tiny_urpc.py").read_text()
    base = base.replace("/root/autodl-tmp/data/yolox_urpc", str(ROOT / "data/yolox"))
    base = base.replace("self.nmsthre = 0.5", f"self.nmsthre = 0.5\n        self.seed = {args.seed}\n        self.output_dir = '{ROOT / 'runs'}'\n        self.save_history_ckpt = False")
    exp_path.write_text(base)
    cmd = [PYTHON, "tools/train.py", "-f", str(exp_path), "-d", "1", "-b", "64",
           "--fp16", "--experiment-name", run_name]
    with (run / "train.log").open("w") as log:
        subprocess.run(cmd, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
    checkpoint = run / "best_ckpt.pth"
    if not checkpoint.exists():
        checkpoint = run / "latest_ckpt.pth"
    assert checkpoint.exists(), checkpoint
    evaluate(checkpoint, args.seed, run)


if __name__ == "__main__":
    main()
