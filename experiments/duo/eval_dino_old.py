"""Eval DINO-4scale R50 (URPC 416 finetune) on test2018 with project protocol:
conf 0.05 / top-100 per image, pycocotools via htdet.engine.evaluator.coco_eval.
Uses DINO's own val transform (416 fixed) and postprocessor (num_select=300, NMS-free).
"""
import json
import os
import sys

import torch

sys.path.insert(0, "/root/autodl-tmp/DINO")
sys.path.insert(0, "/root/autodl-tmp/htdet")

from main import get_args_parser, build_model_main
from util.slconfig import SLConfig
from datasets.coco import build as build_coco_dataset
from htdet.engine.evaluator import coco_eval

CKPT = os.environ.get("DINO_CKPT", "/root/autodl-tmp/work_dirs/dino_4scale_urpc_416_12ep/checkpoint.pth")
OUT = os.environ.get("DINO_OUT", "/root/autodl-tmp/work_dirs/dino_4scale_urpc_416_12ep/eval_pycoco_last.json")
ANN = "/root/autodl-tmp/data/annotations/test_clean.json"


def main():
    sys.argv = ["eval", "-c", "config/DINO/urpc_4scale_416.py",
                "--coco_path", "/root/autodl-tmp/data/dino_urpc_test"]
    args = get_args_parser().parse_args()
    cfg = SLConfig.fromfile(args.config_file)
    for k, v in cfg._cfg_dict.to_dict().items():
        if k not in vars(args):
            setattr(args, k, v)
    if not getattr(args, "use_ema", None):
        args.use_ema = False

    model, criterion, postprocessors = build_model_main(args)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    print("loaded", CKPT, "| epoch:", ck.get("epoch"))
    model.cuda().eval()
    post = postprocessors["bbox"]

    ds = build_coco_dataset("val", args)
    print("test images:", len(ds))

    detections = []
    with torch.no_grad():
        for i in range(len(ds)):
            img, target = ds[i]
            outputs = model(img[None].cuda())
            orig_size = target["orig_size"].unsqueeze(0).cuda()
            res = post(outputs, orig_size)[0]
            boxes = res["boxes"].cpu()
            scores = res["scores"].cpu()
            labels = res["labels"].cpu()
            keep = scores >= 0.05
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if len(scores) == 0:
                continue
            order = torch.argsort(scores, descending=True)[:100]
            detections.append((int(target["image_id"]), boxes[order], scores[order], labels[order]))

    print("images with dets:", len(detections), "/", len(ds))
    ev = coco_eval(detections, ANN)
    print(json.dumps(ev, indent=2))
    json.dump(ev, open(OUT, "w"), indent=2)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
