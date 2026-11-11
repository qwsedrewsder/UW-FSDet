"""S6 B1: offline SNR-quality tagging for train GT boxes.

For each GT box in ORIGINAL image coords (before any resize):
  area     = w * h
  contrast = std(gray(original image box region))     # 0-255 gray std

Tags:
  teacher : area >= 96^2  and  contrast >= class contrast 60th percentile
  student : area <  32^2  or   contrast <  class contrast 30th percentile
  neutral : otherwise (excluded)
If a class has < 100 teachers, lower its area threshold to 64^2.

Writes quality_tag back into train_clean.json (new file train_clean_tagged.json).
"""
import json
import os
import sys

import cv2
import numpy as np

ANN = "/root/autodl-tmp/data/annotations/train_clean.json"
IMG_ROOT = "/root/autodl-tmp/data/urpc_coco/train2018/image"
OUT = "/root/autodl-tmp/data/annotations/train_clean_tagged.json"


def main():
    with open(ANN) as f:
        data = json.load(f)
    id2img = {i["id"]: i for i in data["images"]}

    # collect per-class contrast values
    class_contrast = {c["id"]: [] for c in data["categories"]}
    class_area = {c["id"]: [] for c in data["categories"]}
    cache = {}
    for a in data["annotations"]:
        img = id2img[a["image_id"]]
        if img["id"] not in cache:
            p = os.path.join(IMG_ROOT, img["file_name"])
            im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            cache[img["id"]] = im
        x, y, w, h = a["bbox"]
        roi = cache[img["id"]][max(0, int(y)):int(y + h), max(0, int(x)):int(x + w)]
        if roi.size == 0:
            contrast = 0.0
        else:
            contrast = float(np.std(roi))
        class_contrast[a["category_id"]].append(contrast)
        class_area[a["category_id"]].append(w * h)

    # thresholds
    thr = {}
    for cid in class_contrast:
        cs = np.array(class_contrast[cid])
        a = np.array(class_area[cid])
        c60 = float(np.percentile(cs, 60))
        c30 = float(np.percentile(cs, 30))
        c50 = float(np.percentile(cs, 50))
        n_teacher = int(((a >= 96 * 96) & (cs >= c60)).sum())
        area_thr = 96 * 96 if n_teacher >= 100 else 64 * 64
        thr[cid] = {"c60": c60, "c30": c30, "c50": c50, "area_thr": area_thr}

    # tag
    counts = {"teacher": 0, "student": 0, "neutral": 0}
    per_class = {c["id"]: {"teacher": 0, "student": 0, "neutral": 0} for c in data["categories"]}
    for a in data["annotations"]:
        x, y, w, h = a["bbox"]
        area = w * h
        img = id2img[a["image_id"]]
        roi = cache[img["id"]][max(0, int(y)):int(y + h), max(0, int(x)):int(x + w)]
        contrast = float(np.std(roi)) if roi.size else 0.0
        t = thr[a["category_id"]]
        if 32 * 32 <= area < t["area_thr"] and contrast >= t["c60"]:
            # S6-v3 fix: teacher = MEDIUM objects (32^2..96^2) with high contrast.
            # Excluding large objects (>=96^2) decouples "clear" from "large":
            # v1/v2 prototypes encoded scale, which biased student features
            # toward larger boxes (APm/APl drop via box enlargement).
            tag = "teacher"
        elif area < 32 * 32:
            # student = strictly small (area < 32^2), no contrast condition.
            # All small objects are weak targets worth mimicking.
            tag = "student"
        else:
            tag = "neutral"
        a["quality_tag"] = tag
        counts[tag] += 1
        per_class[a["category_id"]][tag] += 1

    with open(OUT, "w") as f:
        json.dump(data, f)

    print("thresholds:", thr)
    print("totals:", counts)
    for cid, v in per_class.items():
        print(f"class {cid}: {v}")


if __name__ == "__main__":
    main()
