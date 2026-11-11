"""RF-DETR-Nano, backbone-only pretraining (option A), 24 epochs, paper recipe.

- pretrain_weights: DINOv2 ViT-S backbone + random head (rfdetr_nano_dinov2bb.pt)
- batch 32 x grad-accum 4 = effective 128 (paper value)
- lr 1e-4 / encoder 1.5e-4, EMA 0.993, no warmup, no decay,
  layer-wise decay 0.8, grad clip 0.1, hflip + random crop only
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rfdetr import RFDETRNano


def main():
    model = RFDETRNano(
        pretrain_weights="/root/autodl-tmp/htdet/rfdetr_nano_dinov2bb.pt",
        trust_checkpoint=True,
    )
    model.train(
        dataset_dir="/root/autodl-tmp/data/rfdetr_urpc",
        output_dir="/root/autodl-tmp/work_dirs/rfdetr_nano_d2bb_24ep",
        epochs=24,
        lr=1e-4,
        lr_encoder=1.5e-4,
        lr_drop=24,
        batch_size=32,
        grad_accum_steps=4,
        num_workers=12,
        device="cuda",
        eval_interval=1,
        run_test=True,
        checkpoint_interval=8,
        seed=0,
        notes="RF-DETR-Nano DINOv2-backbone-only 24ep URPC2018",
    )


if __name__ == "__main__":
    main()
