"""Apply the shared score threshold and COCO metrics to captured D-FINE boxes."""

import argparse
import json
import sys
import numpy as np
import torch
sys.path.insert(0, "/root/autodl-tmp/htdet")
from htdet.engine.evaluator import coco_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--ann", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    dets = []
    for item in json.loads(open(args.pred).read()):
        b = np.asarray(item["boxes"], dtype=np.float32)
        s = np.asarray(item["scores"], dtype=np.float32)
        l = np.asarray(item["labels"], dtype=np.int64)
        keep = s >= 0.05
        b, s, l = b[keep], s[keep], l[keep]
        if len(s):
            ids = np.argsort(s)[::-1][:100]
            dets.append((item["image_id"], torch.tensor(b[ids]), torch.tensor(s[ids]), torch.tensor(l[ids])))
    metrics = coco_eval(dets, args.ann)
    open(args.out, "w").write(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
