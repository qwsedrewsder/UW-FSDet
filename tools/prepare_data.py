#!/usr/bin/env python
"""S0-C: clean URPC2018 COCO annotations.

Reads instances_{train,test}2018.json, writes {train,test}_clean.json.
Never modifies originals. Produces data_report.md with stats.
"""
import argparse
import json
import os
from collections import Counter

IMG_DIRS = {
    "train": "train2018/image",
    "test": "test2018/image",
}
AREA_BUCKETS = [(0, 32 * 32), (32 * 32, 96 * 96), (96 * 96, float("inf"))]
AREA_NAMES = ["S", "M", "L"]


def bucket_area(area):
    for name, (lo, hi) in zip(AREA_NAMES, AREA_BUCKETS):
        if lo <= area < hi:
            return name
    return "L"


def clean_split(json_path, image_dir, out_path):
    data = json.load(open(json_path))
    imgs = data["images"]
    anns = data["annotations"]
    cats = data["categories"]

    id_to_img = {}
    for img in imgs:
        fn = os.path.basename(img["file_name"])
        img["file_name"] = fn
        id_to_img[img["id"]] = img

    missing = []
    for img in imgs:
        full = os.path.join(image_dir, img["file_name"])
        if not os.path.exists(full):
            missing.append(img["file_name"])

    cat_names = {c["id"]: c["name"] for c in cats}
    before = len(anns)
    dropped_degenerate = 0
    dropped_after_clip = 0
    cat_counts_before = Counter()
    cat_counts_after = Counter()
    area_counts = Counter()
    kept = []
    sizes = Counter()
    # COCOeval treats gt['id']==0 as "unmatched" sentinel; renumber ids from 1
    new_aid = 1

    for ann in anns:
        cat_counts_before[cat_names[ann["category_id"]]] += 1
        img = id_to_img[ann["image_id"]]
        w, h = img["width"], img["height"]
        x, y, bw, bh = [float(v) for v in ann["bbox"]]
        if bw <= 1 or bh <= 1:
            dropped_degenerate += 1
            continue
        x2, y2 = x + bw, y + bh
        x, y = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(w), x2), min(float(h), y2)
        bw, bh = x2 - x, y2 - y
        if bw <= 1 or bh <= 1:
            dropped_after_clip += 1
            continue
        ann = dict(ann)
        ann["id"] = new_aid
        new_aid += 1
        ann["bbox"] = [x, y, bw, bh]
        ann["area"] = bw * bh
        ann.pop("segmentation", None)
        ann["iscrowd"] = 0
        kept.append(ann)
        cat_counts_after[cat_names[ann["category_id"]]] += 1
        area_counts[bucket_area(ann["area"])] += 1

    for img in imgs:
        sizes[(img["width"], img["height"])] += 1

    clean = {"images": imgs, "type": "instances", "categories": cats, "annotations": kept}
    with open(out_path, "w") as f:
        json.dump(clean, f)

    return {
        "split": os.path.basename(json_path),
        "images_before": len(imgs),
        "images_after": len(imgs),
        "anns_before": before,
        "anns_after": len(kept),
        "dropped_degenerate": dropped_degenerate,
        "dropped_after_clip": dropped_after_clip,
        "cat_counts_before": dict(cat_counts_before),
        "cat_counts_after": dict(cat_counts_after),
        "area_counts": dict(area_counts),
        "sizes": dict(sizes),
        "missing_files": missing,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/root/autodl-tmp/data/urpc_coco")
    ap.add_argument("--out-ann", default="/root/autodl-tmp/data/annotations")
    ap.add_argument("--out-report", default="/root/autodl-tmp/work_dirs/data_report.md")
    args = ap.parse_args()

    os.makedirs(args.out_ann, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_report), exist_ok=True)

    lines = []
    for split in ["train", "test"]:
        src = os.path.join(args.data_root, "annotations", f"instances_{split}2018.json")
        img_dir = os.path.join(args.data_root, IMG_DIRS[split])
        dst = os.path.join(args.out_ann, f"{split}_clean.json")
        stats = clean_split(src, img_dir, dst)
        lines.append(f"## {split}\n")
        lines.append(f"- Images: {stats['images_before']} (unchanged by cleaning)")
        lines.append(f"- Bounding boxes: {stats['anns_before']} -> {stats['anns_after']}")
        lines.append(f"- Degenerate boxes discarded: {stats['dropped_degenerate']} + degenerate after clipping: {stats['dropped_after_clip']}")
        lines.append(f"- Boxes per category (before/after): {stats['cat_counts_before']} / {stats['cat_counts_after']}")
        lines.append(f"- COCO area buckets S/M/L: {stats['area_counts']}")
        lines.append(f"- Image size distribution: {stats['sizes']}")
        lines.append(f"- Missing files: {len(stats['missing_files'])} -> {stats['missing_files'][:10]}")
        lines.append("")

    with open(args.out_report, "w") as f:
        f.write("# data_report.md\n\n")
        f.write("\n".join(lines))

    print("\n".join(lines))


if __name__ == "__main__":
    main()
