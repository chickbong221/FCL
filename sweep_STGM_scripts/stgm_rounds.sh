#!/bin/bash
# Sweep for: stgm_rounds

python system/main.py --cfp ./hparams/sweep_STGM/stgm_rounds/stgm_rounds_100.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/stgm_rounds/stgm_rounds_20.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/stgm_rounds/stgm_rounds_40.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/stgm_rounds/stgm_rounds_60.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/stgm_rounds/stgm_rounds_80.json --wandb True