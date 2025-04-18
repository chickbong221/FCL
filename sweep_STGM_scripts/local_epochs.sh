#!/bin/bash
# Sweep for: local_epochs

python system/main.py --cfp ./hparams/sweep_STGM/local_epochs/local_epochs_1.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/local_epochs/local_epochs_10.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/local_epochs/local_epochs_3.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/local_epochs/local_epochs_6.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/local_epochs/local_epochs_8.json --wandb True