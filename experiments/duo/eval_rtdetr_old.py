"""Evaluate RT-DETR-R50 (URPC fine-tuned) on test2018 with the project
protocol: conf 0.05 / NMS 0.5 (RT-DETR is NMS-free; top-100 per image),
pycocotools via htdet.engine.evaluator.coco_eval.

Uses EMA weights from checkpoint.pth (official evaluation convention).
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/RT-DETR/rtdetr_pytorch")
sys.path.insert(0, "/root/autodl-tmp/htdet")

from src.core import YAMLConfig
from src.misc import dist
from htdet.engine.evaluator import coco_eval

CFG = "/root/autodl-tmp/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_urpc_416_nozoom_test.yml"
CKPT = "/root/autodl-tmp/work_dirs/rtdetr_r50vd_urpc_416_nozoom/checkpoint.pth"
ANN = "/root/autodl-tmp/data/annotations/test_clean.json"
OUT = "/root/autodl-tmp/work_dirs/rtdetr_r50vd_urpc_416_nozoom/eval_pycoco_raw.json"


def main():
    dist.init_distributed()
    cfg = YAMLConfig(CFG)
    model = cfg.model
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    ema = state.get("ema")
    if isinstance(ema, dict) and "module" in ema:
        sd = ema["module"]
        print("using EMA weights |", "epoch:", state.get("last_epoch"))
    else:
        sd = state["model"]
        print("using raw model weights |", "epoch:", state.get("last_epoch"))
    model.load_state_dict(sd)
    model.cuda().eval()
    post = cfg.postprocessor
    loader = cfg.val_dataloader

    dets = []
    with torch.no_grad():
        for samples, targets in loader:
            samples = samples.cuda()
            outputs = model(samples)
            orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).cuda()
            results = post(outputs, orig_sizes)
            for i, t in enumerate(targets):
                r = results[i]
                b = r["boxes"].cpu().numpy()
                s = r["scores"].cpu().numpy()
                l = r["labels"].cpu().numpy()
                keep = s >= 0.05
                b, s, l = b[keep], s[keep], l[keep]
                if len(s) == 0:
                    continue
                order = np.argsort(s)[::-1][:100]
                dets.append((int(t["image_id"].item()),
                             torch.tensor(b[order]).float(),
                             torch.tensor(s[order]).float(),
                             torch.tensor(l[order]).long()))
    ev = coco_eval(dets, ANN)
    print(json.dumps(ev, indent=2))
    json.dump(ev, open(OUT, "w"), indent=2)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
