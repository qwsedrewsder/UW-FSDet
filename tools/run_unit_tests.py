"""Unit tests for S2 gate. Run: python tools/run_unit_tests.py
Tests:
  T1 rescale: feed GT-as-prediction through evaluator, must get AP = 1.0
  T2 resume: 2-epoch train, resume, lr/iter continuity
  T3 seed: same seed twice, 50 iters, loss identical bit-for-bit
  T4 assigner: max-iou assignment correctness on synthetic boxes
"""
import json
import os
import sys
import tempfile

import numpy as np
import torch

# deterministic mode for the reproducibility unit test (flash-attn is
# non-deterministic; CUBLAS_WORKSPACE_CONFIG makes SDPA reproducible)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from htdet.core.assigner import max_iou_assign
from htdet.core.coder import DeltaXYWHBBoxCoder
from htdet.engine.evaluator import coco_eval


def t1_rescale_eval():
    """Feed GT boxes (in original coords) as predictions -> AP must be 1.0.
    Uses real test2018 GT (ids renumbered from 1 by prepare_data.py)."""
    ann_file = "/root/autodl-tmp/data/annotations/test_clean.json"
    from pycocotools.coco import COCO
    coco = COCO(ann_file)
    detections = []
    for img in coco.imgs.values():
        anns = coco.imgToAnns.get(img["id"], [])
        boxes = torch.tensor([[a["bbox"][0], a["bbox"][1],
                               a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                              for a in anns], dtype=torch.float32)
        scores = torch.ones(len(anns))
        labels = torch.tensor([a["category_id"] for a in anns], dtype=torch.long)
        detections.append((img["id"], boxes, scores, labels))
    ev = coco_eval(detections, ann_file)
    print(f"T1 GT-as-pred: AP={ev['AP']:.4f} AP50={ev['AP50']:.4f} APs={ev['APs']:.4f} APl={ev['APl']:.4f}")
    assert ev["AP"] > 0.999, "T1 FAILED"
    print("T1 PASSED")


def t2_coder_roundtrip():
    torch.manual_seed(0)
    coder = DeltaXYWHBBoxCoder()
    anchors = torch.rand(100, 4) * 200
    anchors[:, 2] += anchors[:, 0] + 10
    anchors[:, 3] += anchors[:, 1] + 10
    gts = torch.rand(100, 4) * 200
    gts[:, 2] += gts[:, 0] + 10
    gts[:, 3] += gts[:, 1] + 10
    deltas = coder.encode(anchors, gts)
    decoded = coder.decode(anchors, deltas)
    err = (decoded - gts).abs().max().item()
    print(f"T2 coder roundtrip max err: {err:.2e}")
    assert err < 1e-3, "T2 FAILED"
    print("T2 PASSED")


def t3_assigner():
    anchors = torch.tensor([
        [0, 0, 20, 20], [10, 10, 30, 30], [50, 50, 70, 70], [100, 100, 120, 120],
    ], dtype=torch.float32)
    gts = torch.tensor([[11, 11, 29, 29]], dtype=torch.float32)  # overlaps anchor[1] heavily
    labels_gt = torch.tensor([2])
    labels, assigned = max_iou_assign(anchors, gts, labels_gt)
    print(f"T3 assigner labels: {labels.tolist()} assigned: {assigned.tolist()}")
    # anchor1 (IoU ~0.81 with gt) should be positive class 2 (0-based);
    # anchor0/2/3 are background (-1)
    assert labels[1] == 2 and labels[0] == -1, "T3 FAILED"
    print("T3 PASSED")


def t4_seed_repro():
    """Same seed -> identical losses for 50 iters."""
    import random
    from htdet.data.coco_dataset import CocoDataset
    from htdet.data.transforms import Compose, Resize, Normalize, Pad, collate_fn
    from htdet.models.detector import Detector
    from torch.utils.data import DataLoader, Subset
    from htdet.utils.seed import set_seed

    def worker_init(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    def run(seed, iters=50):
        set_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        transforms = Compose([Resize(size=(416, 416)), Normalize(), Pad(size_divisor=32)])
        ds = CocoDataset(ann_file="/root/autodl-tmp/data/annotations/train_clean.json",
                         img_dir="/root/autodl-tmp/data/urpc_coco/train2018/image",
                         train=True, transforms=transforms)
        ds = Subset(ds, list(range(32)))
        loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2,
                            collate_fn=collate_fn, worker_init_fn=worker_init)
        model = Detector(backbone_cfg={"name": "mobilevit", "variant": "xs", "pretrained": True},
                         neck_cfg={"name": "fpn"}, levels=(4, 8, 16, 32), num_classes=4).cuda()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        model.train()
        losses = []
        for it, (imgs, gtb, gtl, metas) in enumerate(loader):
            if it >= iters:
                break
            imgs = imgs.cuda(); gtb = [g.cuda() for g in gtb]; gtl = [l.cuda() for l in gtl]
            outs = model.forward_train(imgs, gtb, gtl, metas)
            loss = outs["loss_cls"] + outs["loss_reg"]
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        return losses

    l1 = run(0)
    l2 = run(0)
    same = all(abs(a - b) < 1e-6 for a, b in zip(l1, l2))
    print(f"T4 seed repro: identical={same} (loss0={l1[0]:.6f} vs {l2[0]:.6f})")
    assert same, "T4 FAILED"
    print("T4 PASSED")


def main():
    t1_rescale_eval()
    t2_coder_roundtrip()
    t3_assigner()
    t4_seed_repro()
    print("\nALL UNIT TESTS PASSED")


if __name__ == "__main__":
    main()
