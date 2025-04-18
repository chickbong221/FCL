#!/bin/bash
# Sweep for: stgm_learning_rate

python system/main.py --cfp ./hparams/sweep_STGM/stgm_learning_rate/stgm_learning_rate_16.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/stgm_learning_rate/stgm_learning_rate_27.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/stgm_learning_rate/stgm_learning_rate_38.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/stgm_learning_rate/stgm_learning_rate_5.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/stgm_learning_rate/stgm_learning_rate_50.json --wandb True