"""S6: aggregate all work_dirs eval.json into REPORT.md + summary table."""
import csv
import json
import os

BASE = "/root/autodl-tmp/work_dirs"


def load_eval(exp):
    p = os.path.join(BASE, exp, "eval.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def main():
    csv_path = os.path.join(BASE, "EXPERIMENTS.csv")
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                rows.append(r)

    lines = ["# HTDet Reproduction Summary\n"]
    lines.append(f"Experiment records: {len(rows)}\n")
    lines.append("| exp | seed | AP | AP50 | AP75 | APs | APm | APl | params | Elapsed time (s) |\n|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['exp']} | {r['seed']} | {r['AP']} | {r['AP50']} | {r['AP75']} | {r['APs']} | {r['APm']} | {r['APl']} | {r['params']} | {r['time_s']} |")

    lines.append("\n## Comparison with the Paper\n")
    paper = {
        "m0": {"AP": 32.2, "AP50": 65.3, "APs": 9.2},
        "m1": {"AP": 37.3, "AP50": 74.6, "APs": 21.9},
        "m2": {"AP": 38.1, "AP50": 75.4, "APs": 22.3},
        "m3": {"AP": 38.5, "AP50": 76.3, "APs": 22.8},
    }
    for key, vals in paper.items():
        lines.append(f"- {key}: paper AP={vals['AP']}; see the corresponding eval.json for reproduced results")

    with open(os.path.join(BASE, "REPORT.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"REPORT written: {os.path.join(BASE, 'REPORT.md')} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
