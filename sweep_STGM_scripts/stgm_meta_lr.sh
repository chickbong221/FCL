#!/bin/bash
# Sweep for: stgm_meta_lr

python system/main.py --cfp ./hparams/sweep_STGM/stgm_meta_lr/stgm_meta_lr_0.1.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/stgm_meta_lr/stgm_meta_lr_0.32.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/stgm_meta_lr/stgm_meta_lr_0.55.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/stgm_meta_lr/stgm_meta_lr_0.78.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/stgm_meta_lr/stgm_meta_lr_1.0.json --wandb True