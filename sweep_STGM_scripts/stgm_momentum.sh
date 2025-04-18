#!/bin/bash
# Sweep for: stgm_momentum

python system/main.py --cfp ./hparams/sweep_STGM/stgm_momentum/stgm_momentum_0.1.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/stgm_momentum/stgm_momentum_0.32.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/stgm_momentum/stgm_momentum_0.55.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/stgm_momentum/stgm_momentum_0.78.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/stgm_momentum/stgm_momentum_1.0.json --wandb True