import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from htdet.data.coco_dataset import CocoDataset
from htdet.data.transforms import collate_fn
from htdet.models.detector import Detector
from torch.utils.data import DataLoader

cfg_path = "configs/d9_r2_ppm.yaml"
import yaml
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
with open(os.path.join("configs", cfg["_base_"])) as f:
    base = yaml.safe_load(f)
for k, v in cfg.items():
    if k != "_base_":
        base.setdefault(k, {}) if isinstance(v, dict) else base.__setitem__(k, v)
        if isinstance(v, dict):
            base[k].update(v)
cfg = base

dcfg, tcfg, mcfg = cfg["data"], cfg["train"], cfg["model"]

train_ds = CocoDataset(
    ann_file=dcfg["ann_train"], img_dir=dcfg["img_train"], train=True,
    normalize_mean=dcfg.get("normalize_mean", [0, 0, 0]),
    normalize_std=dcfg.get("normalize_std", [1, 1, 1]),
    input_size=tuple(dcfg.get("input_size", [416, 416])),
    cache=None, multi_scale=None, color_jitter=False,
)
loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,
                    num_workers=tcfg.get("num_workers", 12), collate_fn=collate_fn,
                    drop_last=True, persistent_workers=True, prefetch_factor=4)

model = Detector(backbone_cfg=mcfg["backbone"], neck_cfg=mcfg.get("neck", {}),
                 levels=tuple(mcfg["levels"]), num_classes=dcfg["num_classes"],
                 pretrained=True, ppm_cfg=mcfg.get("ppm")).cuda()
model.train()
model.apply_ppm_stage(0)

opt = torch.optim.SGD([{"params": [p for p in model.parameters() if p.requires_grad]}],
                      lr=0.05, momentum=0.9, weight_decay=1e-4)

it = iter(loader)
# warmup 2 iters
for _ in range(2):
    imgs, gt_bboxes, gt_labels, metas = next(it)
    imgs = imgs.cuda()
    gt_bboxes = [g.cuda() for g in gt_bboxes]
    gt_labels = [l.cuda() for l in gt_labels]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        losses = model.forward_train(imgs, gt_bboxes, gt_labels, metas)
        loss = losses["loss_cls"] + losses["loss_reg"] + losses["loss_uie"] + losses["loss_sr"]
    loss.backward()
    opt.step()
    opt.zero_grad()
torch.cuda.synchronize()

# data-only timing
t0 = time.time()
for _ in range(5):
    imgs, gt_bboxes, gt_labels, metas = next(it)
torch.cuda.synchronize()
print("data-only: %.3f s/iter" % ((time.time() - t0) / 5))

from torch.profiler import profile, ProfilerActivity

t0 = time.time()
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    for _ in range(5):
        imgs, gt_bboxes, gt_labels, metas = next(it)
        imgs = imgs.cuda()
        gt_bboxes = [g.cuda() for g in gt_bboxes]
        gt_labels = [l.cuda() for l in gt_labels]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            losses = model.forward_train(imgs, gt_bboxes, gt_labels, metas)
            loss = losses["loss_cls"] + losses["loss_reg"] + losses["loss_uie"] + losses["loss_sr"]
        loss.backward()
        opt.step()
        opt.zero_grad()
torch.cuda.synchronize()
print("full step: %.3f s/iter" % ((time.time() - t0) / 5))

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
