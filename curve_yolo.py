import csv

rows = list(csv.reader(open("/root/autodl-tmp/work_dirs/yolov8n_urpc/results.csv")))[1:]
for r in rows:
    if r[0] in ("30", "40", "50", "60", "70", "80", "90", "100"):
        print(f"ep {r[0]:>3}: val mAP50-95 = {r[8]}")
