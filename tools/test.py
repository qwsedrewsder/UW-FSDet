"""python tools/test.py --config <work_dir>/config.yaml --ckpt <path> [--tta]
Evaluates on test2018 and writes eval.json into the config's work_dir."""
import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from htdet.data.coco_dataset import CocoDataset
from htdet.data.transforms import collate_fn, Compose, Resize, Normalize, Pad
from htdet.engine.evaluator import coco_eval
from htdet.models.detector import Detector
from htdet.utils.checkpoint import load_checkpoint


def build_test_loader(cfg, keep_ratio=False):
    dcfg = cfg["data"]
    size = tuple(dcfg.get("test", {}).get("input_size", dcfg.get("input_size", [416, 416])))
    transforms = Compose([
        Resize(size=size, keep_ratio=keep_ratio),
        Normalize(mean=dcfg.get("normalize_mean", [0, 0, 0]), std=dcfg.get("normalize_std", [1, 1, 1])),
        Pad(size_divisor=32),
    ])
    ds = CocoDataset(
        ann_file=dcfg["ann_test"], img_dir=dcfg["img_test"], train=False, transforms=transforms,
    )
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to work_dir config.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--tta-scales", default="320,416", help="comma list of long-side scales")
    ap.add_argument("--tta-flip", action="store_true")
    ap.add_argument("--out", default=None, help="eval.json output path")
    ap.add_argument("--score-thr", type=float, default=None, help="override score_thr (default from config 0.05)")
    ap.add_argument("--nms-iou", type=float, default=None, help="override NMS IoU threshold (default 0.5)")
    ap.add_argument("--soft-nms", action="store_true", help="use gaussian soft-NMS")
    ap.add_argument("--max-per-img", type=int, default=None, help="override max_per_img")
    ap.add_argument("--use-ema", action="store_true", help="evaluate EMA weights if saved")
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    work_dir = os.path.dirname(args.config)

    torch.backends.cudnn.benchmark = True

    mcfg = cfg["model"]
    model = Detector(
        backbone_cfg=mcfg["backbone"],
        neck_cfg=mcfg.get("neck", {}),
        levels=tuple(mcfg["levels"]),
        num_classes=cfg["data"]["num_classes"],
        pretrained=False,
        ppm_cfg=mcfg.get("ppm", None),
        head_cfg=mcfg.get("head", None),
    )
    state = load_checkpoint(args.ckpt)
    if args.use_ema and "ema_model" in state:
        model.load_state_dict(state["ema_model"])
        print("loaded EMA weights")
    else:
        model.load_state_dict(state["model"])
    model.cuda().eval()

    nms_pre = cfg["postprocess"].get("nms_pre", 1000)
    score_thr = args.score_thr if args.score_thr is not None else cfg["postprocess"].get("score_thr", 0.05)
    nms_iou = args.nms_iou if args.nms_iou is not None else cfg["postprocess"].get("nms_iou", 0.5)
    max_per_img = args.max_per_img if args.max_per_img is not None else cfg["postprocess"].get("max_per_img", 100)
    soft_nms = args.soft_nms

    if not args.tta:
        loader = build_test_loader(cfg, keep_ratio=False)
        detections = []
        import numpy as np
        import torchvision
        with torch.no_grad():
            for imgs, _, _, metas in loader:
                outs = model.forward_test(imgs.cuda(), metas, nms_pre=nms_pre, score_thr=score_thr,
                                          max_per_img=max_per_img, nms_iou=nms_iou, soft_nms=soft_nms)
                for o, meta in zip(outs, metas):
                    detections.append((meta["img_id"], o[:, :4], o[:, 4], o[:, 5].long()))
        ev = coco_eval(detections, cfg["data"]["ann_test"])
    else:
        from htdet.engine.tta import tta_eval
        ev = tta_eval(model, cfg, scales=[int(s) for s in args.tta_scales.split(",")],
                      flip=args.tta_flip, ann_file=cfg["data"]["ann_test"])

    out = {"AP": ev["AP"], "AP50": ev["AP50"], "AP75": ev["AP75"],
           "APs": ev["APs"], "APm": ev["APm"], "APl": ev["APl"],
           "params": sum(p.numel() for p in model.parameters()),
           "tta": args.tta, "tta_scales": args.tta_scales, "tta_flip": args.tta_flip,
           "ckpt": args.ckpt, "score_thr": score_thr, "nms_iou": nms_iou,
           "soft_nms": soft_nms, "max_per_img": max_per_img}
    out_path = args.out or os.path.join(work_dir, "eval.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
