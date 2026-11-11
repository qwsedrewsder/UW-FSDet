"""Eval DECO R50 (URPC 416 finetune) on test2018 with project protocol:
conf 0.05 / top-100 per image, pycocotools via htdet.engine.evaluator.coco_eval.
"""
import json
import os
import sys

import torch

sys.path.insert(0, "/root/autodl-tmp/DECO/deco")
sys.path.insert(0, "/root/autodl-tmp/htdet")

sys.argv = ["eval", "--backbone", "resnet50",
            "--coco_path", "/root/autodl-tmp/data/dino_urpc_test",
            "--num_classes", "4", "--batch_size", "4"]

from main import get_args_parser
from models import build_model
import models.util.misc as utils
from datasets import build_dataset
from torch.utils.data import DataLoader
from htdet.engine.evaluator import coco_eval

CKPT = os.environ.get("DECO_CKPT", "/root/autodl-tmp/work_dirs/deco_r50_urpc_416_150ep/checkpoint.pth")
OUT = os.environ.get("DECO_OUT", "/root/autodl-tmp/work_dirs/deco_r50_urpc_416_150ep/eval_pycoco_last.json")
ANN = "/root/autodl-tmp/data/annotations/test_clean.json"


def main():
    args, _ = get_args_parser().parse_known_args()
    model, criterion, postprocessors = build_model(args)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    print("loaded", CKPT, "| epoch:", ck.get("epoch"))
    model.cuda().eval()

    ds = build_dataset("val", args)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        drop_last=False, collate_fn=utils.collate_fn, num_workers=4)
    print("test images:", len(ds))

    detections = []
    with torch.no_grad():
        for samples, targets in loader:
            samples = samples.to("cuda")
            outputs = model(samples)
            orig_sizes = torch.stack([t["orig_size"] for t in targets]).cuda()
            results = postprocessors["bbox"](outputs, orig_sizes)
            for t, r in zip(targets, results):
                scores = r["scores"].cpu()
                boxes = r["boxes"].cpu()
                labels = r["labels"].cpu()
                keep = scores >= 0.05
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                if len(scores) == 0:
                    continue
                detections.append((int(t["image_id"]), boxes, scores, labels))

    print("images with dets:", len(detections), "/", len(ds))
    ev = coco_eval(detections, ANN)
    print(json.dumps(ev, indent=2))
    json.dump(ev, open(OUT, "w"), indent=2)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
