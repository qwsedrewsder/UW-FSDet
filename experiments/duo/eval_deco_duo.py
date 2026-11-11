"""Evaluate DECO on DUO test with the shared COCO score filtering."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path("/root/duo_exp")
sys.path.insert(0, "/root/autodl-tmp/DECO/deco")
sys.path.insert(0, "/root/autodl-tmp/htdet")
from main import get_args_parser
from models import build_model
import models.util.misc as utils
from datasets import build_dataset
from htdet.engine.evaluator import coco_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    args_cli = parser.parse_args()
    sys.argv = ["eval", "--backbone", "resnet50", "--coco_path", str(ROOT / "data/dino_test"),
                "--num_classes", "4", "--batch_size", "8"]
    args, _ = get_args_parser().parse_known_args()
    model, _, post = build_model(args)
    state = torch.load(args_cli.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.cuda().eval()
    dataset = build_dataset("val", args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                        collate_fn=utils.collate_fn, num_workers=4)
    detections = []
    with torch.no_grad():
        for samples, targets in loader:
            outputs = model(samples.cuda())
            sizes = torch.stack([target["orig_size"] for target in targets]).cuda()
            for target, result in zip(targets, post["bbox"](outputs, sizes)):
                boxes, scores, labels = result["boxes"].cpu(), result["scores"].cpu(), result["labels"].cpu()
                keep = scores >= 0.05
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                if len(scores):
                    order = torch.argsort(scores, descending=True)[:100]
                    detections.append((int(target["image_id"]), boxes[order], scores[order], labels[order]))
    metrics = coco_eval(detections, str(ROOT / "data/annotations/instances_test.json"))
    Path(args_cli.out).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
