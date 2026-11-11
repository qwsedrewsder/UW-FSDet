import json
from pathlib import Path

models = [
    ("uwfsdet", "htdet", "uwfsdet"),
    ("baseline", "htdet", "baseline"),
    ("retinanet", "htdet", "retinanet"),
    ("yolov12n", "yolov12", "n"),
    ("yolov12s", "yolov12", "s"),
    ("yolov12m", "yolov12", "m"),
    ("yolov12l", "yolov12", "l"),
    ("hyuod", "script", "run_hyuod_duo.py"),
    ("yolox_tiny", "script", "run_yolox_duo.py"),
    ("dfine", "script", "run_dfine_duo.py"),
    ("dino", "script", "run_dino_duo.py"),
    ("rtdetr", "script", "run_rtdetr_duo.py"),
    ("eamsf", "script", "run_eamsf_duo.py"),
    ("deco", "script", "run_deco_duo.py"),
]

jobs = [{"name": "e2_seed0", "kind": "htdet", "model": "e2", "seed": 0}]
for seed in range(3):
    if seed:
        jobs.append({"name": f"e2_seed{seed}", "kind": "htdet", "model": "e2", "seed": seed})
    for name, kind, value in models:
        job = {"name": f"{name}_seed{seed}", "kind": kind, "seed": seed}
        if kind == "script":
            job["script"] = value
        elif kind == "yolov12":
            job["model"] = value
        else:
            job["model"] = value
        jobs.append(job)

Path(r"C:\Users\LYC\duo_runs\schedule.json").write_text(
    json.dumps({"jobs": jobs}, indent=2), encoding="utf-8"
)
print(f"{len(jobs)} queue entries: {len(models) * 3} reported runs + 3 DUO E2 pretrains")
