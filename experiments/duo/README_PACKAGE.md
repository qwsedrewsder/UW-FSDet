# Available comparison experiment scripts

This directory contains the existing Python scripts, YAML configurations, and `schedule.json` from the local `duo_runs` directory. Files containing `_old` in their names are earlier scripts; the original names are retained to distinguish them from later versions.

- `run_*_duo.py`: DUO experiment entry points for individual models.
- `eval_*` and `*_eval_old.py`: existing evaluation scripts.
- `prepare_*.py`: existing data preparation scripts.
- `run_htdet_duo.py` and `htdet_train_duo.py`: experiments for the htdet family, using model implementations in `UW-FSDet/`.
- `queue_duo.py` and `schedule.json`: the existing experiment queue and execution order, provided as a record of the original workflow.

The scripts retain the original server paths, including `/root/duo_exp` and `/root/autodl-tmp`. External projects and initialization checkpoints referenced by these scripts must be prepared separately. Apart from D-FINE, complete source trees for the external comparison models were not found in the checked local locations. This directory does not contain standalone implementations of all comparison models.

No result files, model weights, or datasets are included. The training, evaluation, and queue scripts were not executed during packaging.
