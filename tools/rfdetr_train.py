"""Train RF-DETR-Nano on URPC2018 (Roboflow COCO layout).

Recipe: paper defaults (lr 1e-4, EMA, layer-wise decay, clip 0.1, flip+crop only)
        30 epochs, batch 16 x grad-accum 2 (eff. 32), native 560 resolution.
Dataset: /root/autodl-tmp/data/rfdetr_urpc/{train,valid,test}
Output:  /root/autodl-tmp/work_dirs/rfdetr_nano_urpc
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rfdetr import RFDETRNano


def main():
    model = RFDETRNano()
    model.train(
        dataset_dir="/root/autodl-tmp/data/rfdetr_urpc",
        output_dir="/root/autodl-tmp/work_dirs/rfdetr_nano_urpc",
        epochs=30,
        lr=1e-4,
        lr_encoder=1.5e-4,
        lr_drop=30,
        batch_size=16,
        grad_accum_steps=2,
        num_workers=12,
        device="cuda",
        eval_interval=1,
        run_test=True,
        checkpoint_interval=10,
        seed=0,
        notes="RF-DETR-Nano URPC2018 30ep",
    )


if __name__ == "__main__":
    main()
