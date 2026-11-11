"""Paper-4.6 error analysis (Loc/Oth/BG/FN) for test2018 predictions.

Usage:
  python tools/error_analysis_paper46.py --config <cfg>.yaml --ckpt <final.pth> --out-dir <dir>

Classification rules (paper 4.6, per prediction, greedy TP by score):
  TP  : pred class correct, IoU >= 0.5          (one GT consumed by one TP)
  Loc : pred class correct, 0.1 < IoU < 0.5
  Oth : pred class wrong,   IoU > 0.1           (any IoU > 0.1, incl. >= 0.5)
  BG  : pred IoU < 0.1 with ALL GTs
  FN (per GT, not consumed by any TP):
        sub=miscls : best det IoU > 0.5 but class wrong   (paper literal)
        sub=miss   : best det IoU <= 0.5                  (extension, pure miss)

Outputs (out-dir):
  err_stats.json   counts per {type}_{cls}_{band} + totals (FN split by sub)
  D1_Loc.json / D1_Oth.json / D1_BG.json / D1_FN.json   per-sample lists
  vis/<img_id>.jpg green=GT, red=FN, purple=BG, orange=Loc, yellow=Oth
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from htdet.data.coco_dataset import CocoDataset
from htdet.data.transforms import collate_fn, Compose, Resize, Normalize, Pad
from htdet.models.detector import Detector
from htdet.utils.checkpoint import load_checkpoint

COCO_NAMES = {0: "holothurian", 1: "echinus", 2: "scallop", 3: "starfish"}
ABBR = {0: "holo", 1: "echi", 2: "scal", 3: "star"}
BAND_NAMES = {0: "S", 1: "M", 2: "L"}


def area_band(area):
    if area < 32 * 32:
        return 0
    if area < 96 * 96:
        return 1
    return 2


def build_test_loader(cfg):
    dcfg = cfg["data"]
    size = tuple(dcfg.get("test", {}).get("input_size", dcfg.get("input_size", [416, 416])))
    transforms = Compose([
        Resize(size=size, keep_ratio=False),
        Normalize(mean=dcfg.get("normalize_mean", [0, 0, 0]), std=dcfg.get("normalize_std", [1, 1, 1])),
        Pad(size_divisor=32),
    ])
    ds = CocoDataset(ann_file=dcfg["ann_test"], img_dir=dcfg["img_test"], train=False, transforms=transforms)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)


def detect(model, loader, cfg, pp):
    model.eval()
    dets = []
    with torch.no_grad():
        for imgs, _, _, metas in loader:
            outs = model.forward_test(
                imgs.cuda(), metas,
                nms_pre=pp.get("nms_pre", 1000), score_thr=pp.get("score_thr", 0.05),
                nms_iou=pp.get("nms_iou", 0.5), max_per_img=pp.get("max_per_img", 100))
            for o, meta in zip(outs, metas):
                if o.shape[0]:
                    dets.append((meta["img_id"], o[:, :4].cpu().numpy(),
                                 o[:, 4].cpu().numpy(), o[:, 5].long().cpu().numpy()))
                else:
                    dets.append((meta["img_id"], np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64)))
    return dets


def iou_matrix(dets, gts):
    if len(dets) == 0 or len(gts) == 0:
        return np.zeros((len(dets), len(gts)))
    d = dets
    g = gts
    inter = np.maximum(0, np.minimum(d[:, None, 2], g[None, :, 2]) - np.maximum(d[:, None, 0], g[None, :, 0]))
    inter *= np.maximum(0, np.minimum(d[:, None, 3], g[None, :, 3]) - np.maximum(d[:, None, 1], g[None, :, 1]))
    da = (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1])
    ga = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    union = da[:, None] + ga[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def classify_paper(detections, coco_gt):
    gts_by_img = {}
    for ann in coco_gt.anns.values():
        gts_by_img.setdefault(int(ann["image_id"]), []).append(ann)

    tp = loc = oth = bg = fn = 0
    fn_miscls = fn_miss = 0
    loc_items, oth_items, bg_items, fn_items = [], [], [], []
    per_img = []

    for img_id, dbox, dscore, dcls in detections:
        gts = gts_by_img.get(int(img_id), [])
        gt_boxes = np.array([a["bbox"] for a in gts], dtype=np.float32)
        if gt_boxes.shape[0]:
            gt_boxes = np.stack([gt_boxes[:, 0], gt_boxes[:, 1],
                                 gt_boxes[:, 0] + gt_boxes[:, 2], gt_boxes[:, 1] + gt_boxes[:, 3]], axis=1)
        gt_cls = np.array([a["category_id"] for a in gts], dtype=np.int64)
        gt_areas = np.array([a["bbox"][2] * a["bbox"][3] for a in gts], dtype=np.float32)

        iou = iou_matrix(dbox, gt_boxes)  # (M, N)
        order = np.argsort(-dscore)
        used_gt = set()  # GTs consumed by a TP
        for di in order:
            if iou.shape[1] == 0:
                bg += 1
                bg_items.append({
                    "img_id": int(img_id), "type": "BG", "det_cls": int(dcls[di]),
                    "gt_cls": -1, "band": area_band((dbox[di, 2] - dbox[di, 0]) * (dbox[di, 3] - dbox[di, 1])),
                    "score": round(float(dscore[di]), 4), "iou": 0.0,
                    "det_box": [round(float(v), 1) for v in dbox[di]]})
                continue
            # greedy TP: only against unconsumed GTs
            avail = [g for g in range(len(gts)) if g not in used_gt]
            if avail:
                gi = int(avail[int(np.argmax(iou[di, avail]))])
            else:
                gi = int(np.argmax(iou[di]))
            best_iou_tp = float(iou[di, gi]) if gi not in used_gt else 0.0
            if best_iou_tp >= 0.5 and dcls[di] == gt_cls[gi]:
                tp += 1
                used_gt.add(gi)
                continue
            # error classification against ALL GTs (paper rules)
            gi_all = int(np.argmax(iou[di]))
            iou_all = float(iou[di, gi_all])
            if iou_all < 0.1:
                bg += 1
                bg_items.append({
                    "img_id": int(img_id), "type": "BG", "det_cls": int(dcls[di]),
                    "gt_cls": int(gt_cls[gi_all]), "band": area_band((dbox[di, 2] - dbox[di, 0]) * (dbox[di, 3] - dbox[di, 1])),
                    "score": round(float(dscore[di]), 4), "iou": round(iou_all, 3),
                    "det_box": [round(float(v), 1) for v in dbox[di]]})
            elif dcls[di] != gt_cls[gi_all]:
                oth += 1
                oth_items.append({
                    "img_id": int(img_id), "type": "Oth", "det_cls": int(dcls[di]),
                    "gt_cls": int(gt_cls[gi_all]), "band": area_band(float(gt_areas[gi_all])),
                    "score": round(float(dscore[di]), 4), "iou": round(iou_all, 3),
                    "det_box": [round(float(v), 1) for v in dbox[di]]})
            elif 0.1 < iou_all < 0.5:
                loc += 1
                loc_items.append({
                    "img_id": int(img_id), "type": "Loc", "det_cls": int(dcls[di]),
                    "gt_cls": int(gt_cls[gi_all]), "band": area_band(float(gt_areas[gi_all])),
                    "score": round(float(dscore[di]), 4), "iou": round(iou_all, 3),
                    "det_box": [round(float(v), 1) for v in dbox[di]]})
            # else: same-class IoU>=0.5 but GT already consumed -> duplicate TP, not an error

        # FN per GT: not consumed by any TP
        for gi, a in enumerate(gts):
            if gi in used_gt:
                continue
            if iou.shape[0]:
                bj = int(np.argmax(iou[:, gi]))
                best = float(iou[bj, gi])
            else:
                bj, best = -1, 0.0
            fn += 1
            if best > 0.5 and bj >= 0 and dcls[bj] != gt_cls[gi]:
                sub = "miscls"
                fn_miscls += 1
            else:
                sub = "miss"
                fn_miss += 1
            fn_items.append({
                "img_id": int(img_id), "type": "FN", "det_cls": -1,
                "gt_cls": int(gt_cls[gi]), "band": area_band(float(gt_areas[gi])),
                "score": 0.0, "iou": round(best, 3), "sub": sub,
                "gt_box": [round(float(v), 1) for v in gt_boxes[gi]]})

        per_img.append({
            "img_id": int(img_id), "n_dets": int(len(dbox)), "n_gt": int(len(gts)),
            "scores": [round(float(s), 4) for s in np.sort(dscore)[::-1][:100]]})

    stats = {}
    for typ, items in [("FN", fn_items), ("BG", bg_items), ("Loc", loc_items), ("Oth", oth_items)]:
        for it in items:
            cls_key = it["gt_cls"] if it["gt_cls"] >= 0 else it["det_cls"]
            key = "%s_%s_band%d" % (typ, COCO_NAMES[cls_key], it["band"])
            stats[key] = stats.get(key, 0) + 1
    stats["_totals"] = {"TP": tp, "Loc": loc, "Oth": oth, "BG": bg,
                        "FN": fn, "FN_miscls": fn_miscls, "FN_miss": fn_miss}
    stats["_per_img"] = per_img
    return stats, loc_items, oth_items, bg_items, fn_items


def visualize_paper(detections, coco_gt, img_dir, out_dir):
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    gts_by_img = {}
    for ann in coco_gt.anns.values():
        gts_by_img.setdefault(int(ann["image_id"]), []).append(ann)
    name_by_id = {iid: info["file_name"] for iid, info in coco_gt.imgs.items()}

    err_by_img = {}
    for img_id, dbox, dscore, dcls in detections:
        gts = gts_by_img.get(int(img_id), [])
        gt_boxes = np.array([a["bbox"] for a in gts], dtype=np.float32)
        if gt_boxes.shape[0]:
            gt_boxes = np.stack([gt_boxes[:, 0], gt_boxes[:, 1],
                                 gt_boxes[:, 0] + gt_boxes[:, 2], gt_boxes[:, 1] + gt_boxes[:, 3]], axis=1)
        gt_cls = np.array([a["category_id"] for a in gts], dtype=np.int64)
        iou = iou_matrix(dbox, gt_boxes)
        order = np.argsort(-dscore)
        used_gt = set()
        kinds = []
        for di in order:
            if iou.shape[1] == 0:
                kinds.append(("BG", dbox[di], int(dcls[di]), float(dscore[di])))
                continue
            avail = [g for g in range(len(gts)) if g not in used_gt]
            if avail:
                gi = int(avail[int(np.argmax(iou[di, avail]))])
            else:
                gi = int(np.argmax(iou[di]))
            best_iou_tp = float(iou[di, gi]) if gi not in used_gt else 0.0
            if best_iou_tp >= 0.5 and dcls[di] == gt_cls[gi]:
                used_gt.add(gi)
                kinds.append(("TP", dbox[di], int(dcls[di]), float(dscore[di])))
                continue
            gi_all = int(np.argmax(iou[di]))
            iou_all = float(iou[di, gi_all])
            if iou_all < 0.1:
                kinds.append(("BG", dbox[di], int(dcls[di]), float(dscore[di])))
            elif dcls[di] != gt_cls[gi_all]:
                kinds.append(("Oth", dbox[di], int(dcls[di]), float(dscore[di])))
            elif 0.1 < iou_all < 0.5:
                kinds.append(("Loc", dbox[di], int(dcls[di]), float(dscore[di])))
        for gi, a in enumerate(gts):
            if gi in used_gt:
                continue
            if iou.shape[0]:
                bj = int(np.argmax(iou[:, gi]))
                best = float(iou[bj, gi])
            else:
                bj, best = -1, 0.0
            if best > 0.5 and bj >= 0 and dcls[bj] != gt_cls[gi]:
                kinds.append(("FN_miscls", gt_boxes[gi], int(gt_cls[gi]), 0.0))
            else:
                kinds.append(("FN", gt_boxes[gi], int(gt_cls[gi]), 0.0))
        err_by_img[int(img_id)] = kinds

    colors = {"TP": (0, 255, 0), "FN": (0, 0, 255), "FN_miscls": (0, 0, 255),
              "BG": (255, 0, 255), "Loc": (0, 165, 255), "Oth": (0, 255, 255)}
    for iid, info in coco_gt.imgs.items():
        kinds = err_by_img.get(int(iid), [])
        if not kinds:
            continue
        path = os.path.join(img_dir, name_by_id[iid])
        img = cv2.imread(path)
        if img is None:
            continue
        for kind, box, cls_id, score in kinds:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[kind], 2)
            if kind == "TP":
                label = "TP_%s(%.2f)" % (ABBR[cls_id], score)
            elif kind in ("FN", "FN_miscls"):
                label = "%s_%s" % (kind, COCO_NAMES[cls_id])
            else:
                label = "%s%s(%.2f)" % (kind, ABBR[cls_id], score)
            cv2.putText(img, label, (x1, max(0, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[kind], 1)
        cv2.imwrite(os.path.join(out_dir, "%d.jpg" % iid), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--score-thr", type=float, default=None)
    ap.add_argument("--max-per-img", type=int, default=None)
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg["model"]
    model = Detector(backbone_cfg=mcfg["backbone"], neck_cfg=mcfg.get("neck", {}),
                     levels=tuple(mcfg["levels"]), num_classes=cfg["data"]["num_classes"], pretrained=False)
    state = load_checkpoint(args.ckpt)
    model.load_state_dict(state["model"])
    model.cuda().eval()

    pp = dict(cfg.get("postprocess", {}))
    if args.score_thr is not None:
        pp["score_thr"] = args.score_thr
    if args.max_per_img is not None:
        pp["max_per_img"] = args.max_per_img

    loader = build_test_loader(cfg)
    detections = detect(model, loader, cfg, pp)
    coco_gt = COCO(cfg["data"]["ann_test"])
    stats, loc_items, oth_items, bg_items, fn_items = classify_paper(detections, coco_gt)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "err_stats.json"), "w") as f:
        json.dump(stats, f, indent=1, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "dets_stats.json"), "w") as f:
        json.dump({"postprocess": pp, "per_img": stats.pop("_per_img")}, f, ensure_ascii=False)
    for name, items in [("D1_Loc.json", loc_items), ("D1_Oth.json", oth_items),
                        ("D1_BG.json", bg_items), ("D1_FN.json", fn_items)]:
        with open(os.path.join(args.out_dir, name), "w") as f:
            json.dump(items, f, ensure_ascii=False)

    visualize_paper(detections, coco_gt, cfg["data"]["img_test"], os.path.join(args.out_dir, "vis"))
    print(json.dumps(stats.get("_totals", {}), indent=1))
    print("done ->", args.out_dir)


if __name__ == "__main__":
    main()
