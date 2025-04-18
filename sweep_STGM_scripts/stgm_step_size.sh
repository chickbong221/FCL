#!/bin/bash
# Sweep for: stgm_step_size

python system/main.py --cfp ./hparams/sweep_STGM/stgm_step_size/stgm_step_size_10.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/stgm_step_size/stgm_step_size_20.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/stgm_step_size/stgm_step_size_30.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/stgm_step_size/stgm_step_size_40.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/stgm_step_size/stgm_step_size_50.json --wandb True