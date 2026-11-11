"""Preprocess train/test images to cached uint8 416x416 tensors.
Speeds up training (GPU was at 35% util, data loading bottleneck).

python tools/precache.py
Output: /root/autodl-tmp/data/cache/{train,test}_416.npy + meta json
"""
import os
import sys
import time

import cv2
import numpy as np

DATA = "/root/autodl-tmp/data/urpc_coco"
OUT = "/root/autodl-tmp/data/cache"


def precache(split):
    img_dir = os.path.join(DATA, f"{split}2018", "image")
    files = sorted(os.listdir(img_dir))
    os.makedirs(OUT, exist_ok=True)
    arr = np.zeros((len(files), 416, 416, 3), dtype=np.uint8)
    t0 = time.time()
    for i, fn in enumerate(files):
        p = os.path.join(img_dir, fn)
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_LINEAR)
        arr[i] = img
        if i % 500 == 0:
            print(f"{split} {i}/{len(files)} {time.time()-t0:.0f}s", flush=True)
    np.save(os.path.join(OUT, f"{split}_416.npy"), arr)
    with open(os.path.join(OUT, f"{split}_files.json"), "w") as f:
        import json
        json.dump({fn: i for i, fn in enumerate(files)}, f)
    print(f"{split} done: {len(files)} imgs, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    for sp in ["train", "test"]:
        precache(sp)
