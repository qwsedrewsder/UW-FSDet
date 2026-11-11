import os
import sys

import torch

sys.path.insert(0, "/root/autodl-tmp/htdet")

from htdet.models.detector import Detector
from htdet.utils.checkpoint import load_checkpoint

import yaml
with open("/root/autodl-tmp/work_dirs/d9_r2_ppm_seed0/config.yaml") as f:
    cfg = yaml.safe_load(f)

mcfg = cfg["model"]
model = Detector(backbone_cfg=mcfg["backbone"], neck_cfg=mcfg.get("neck", {}),
                 levels=tuple(mcfg["levels"]), num_classes=cfg["data"]["num_classes"],
                 pretrained=False, ppm_cfg=mcfg.get("ppm"))
state = load_checkpoint("/root/autodl-tmp/work_dirs/d9_r2_ppm_seed0/ckpt/final.pth", device="cpu")
model.load_state_dict(state["model"])
model.cuda().eval()

# read a few real train images through the same pipeline as training
from htdet.data.coco_dataset import CocoDataset
ds = CocoDataset(ann_file=cfg["data"]["ann_train"], img_dir=cfg["data"]["img_train"],
                 train=False, normalize_mean=cfg["data"].get("normalize_mean", [0, 0, 0]),
                 normalize_std=cfg["data"].get("normalize_std", [1, 1, 1]),
                 input_size=tuple(cfg["data"].get("input_size", [416, 416])))

for i in [0, 100, 500]:
    img, boxes, labels, meta = ds[i]
    x = img.unsqueeze(0).cuda()
    with torch.no_grad():
        x_pm, _ = model.ppm(x)
        y_uie = model.ppm.uie(x)[1]
        y_sr = model.ppm.sr(x)[1]
        feats = model.backbone(x_pm)
    print(f"img {i}: x [{x.min():.3f},{x.max():.3f}] mean {x.mean():.3f} std {x.std():.3f}")
    print(f"      y_uie [{y_uie.min():.3f},{y_uie.max():.3f}] mean {y_uie.mean():.3f} std {y_uie.std():.3f}")
    print(f"      y_sr  [{y_sr.min():.3f},{y_sr.max():.3f}] mean {y_sr.mean():.3f} std {y_sr.std():.3f}")
    print(f"      x_pm  [{x_pm.min():.3f},{x_pm.max():.3f}] mean {x_pm.mean():.3f} std {x_pm.std():.3f}")
    print(f"      C1 shape {tuple(feats[0].shape)} mean {feats[0].mean():.3f} std {feats[0].std():.3f}")
