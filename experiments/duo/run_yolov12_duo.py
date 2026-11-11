"""Run one YOLOv12 DUO seed using the existing URPC2018 training recipe."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, "/root/autodl-tmp/yolov12")
sys.path.insert(0, "/root/autodl-tmp/htdet")
from ultralytics import YOLO
from htdet.engine.evaluator import coco_eval


ROOT = Path("/root/duo_exp")


def evaluate(checkpoint: Path, output: Path) -> None:
    ann = ROOT / "data/annotations/instances_test.json"
    image_dir = ROOT / "data/raw/DUO/images/test"
    detector = YOLO(str(checkpoint))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw = state.get("model") or state.get("ema")
    if isinstance(raw, torch.nn.Module):
        raw = raw.state_dict()
    detector.model.load_state_dict(raw)
    detector.model.cuda().eval()
    data = json.loads(ann.read_text())
    name_to_id = {image["file_name"]: image["id"] for image in data["images"]}
    files = sorted(name_to_id)
    detections = []
    for start in range(0, len(files), 32):
        current = files[start:start + 32]
        outputs = detector.predict([str(image_dir / f) for f in current], imgsz=416,
                                   conf=0.05, iou=0.5, verbose=False)
        for name, result in zip(current, outputs):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy().astype(np.int64)
            keep = np.argsort(scores)[::-1][:100]
            detections.append((name_to_id[name], torch.tensor(boxes[keep]).float(),
                               torch.tensor(scores[keep]).float(),
                               torch.tensor(labels[keep]).long()))
    metrics = coco_eval(detections, str(ann))
    output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=("n", "s", "m", "l"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    args = parser.parse_args()
    assert (ROOT / "data/manifest.json").exists()
    run_dir = ROOT / "runs" / f"yolov12{args.size}_seed{args.seed}"
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(f"/root/autodl-tmp/yolov12/yolov12{args.size}.pt")
    model.train(
        data=str(ROOT / "data/yolo/data.yaml"), epochs=200, imgsz=416, batch=64,
        optimizer="SGD", lr0=0.01, momentum=0.937, weight_decay=0.0005,
        mosaic=1.0, close_mosaic=10, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        translate=0.1, fliplr=0.5, scale=0.0, mixup=0.0, copy_paste=0.0,
        degrees=0.0, shear=0.0, perspective=0.0, flipud=0.0, erasing=0.0,
        auto_augment=None, device=0, workers=8, project=str(run_dir.parent),
        name=run_dir.name, exist_ok=True, seed=args.seed, patience=0,
        save_period=-1,
    )
    checkpoint = run_dir / "weights/last.pt"
    assert checkpoint.exists(), checkpoint
    evaluate(checkpoint, run_dir / "test_eval.json")


if __name__ == "__main__":
    main()
