"""Draw GT boxes only (green) on all test2018 images.

Usage:
  python tools/draw_gt.py --ann <test_clean.json> --img-dir <dir> --out-dir <dir>
"""
import argparse
import os
import sys

import cv2
from pycocotools.coco import COCO

COCO_NAMES = {0: "holothurian", 1: "echinus", 2: "scallop", 3: "starfish"}
ABBR = {0: "holo", 1: "echi", 2: "scal", 3: "star"}


def area_band(area):
    if area < 32 * 32:
        return "S"
    if area < 96 * 96:
        return "M"
    return "L"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    coco = COCO(args.ann)
    os.makedirs(args.out_dir, exist_ok=True)
    gts_by_img = {}
    for ann in coco.anns.values():
        gts_by_img.setdefault(int(ann["image_id"]), []).append(ann)

    for iid, info in coco.imgs.items():
        img = cv2.imread(os.path.join(args.img_dir, info["file_name"]))
        if img is None:
            print("missing", info["file_name"])
            continue
        for a in gts_by_img.get(int(iid), []):
            x, y, w, h = [int(v) for v in a["bbox"]]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            band = area_band(w * h)
            label = "%s%s" % (ABBR[a["category_id"]], band)
            cv2.putText(img, label, (x, max(0, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(args.out_dir, "%d.jpg" % int(iid)), img)

    print("done ->", args.out_dir)


if __name__ == "__main__":
    main()
