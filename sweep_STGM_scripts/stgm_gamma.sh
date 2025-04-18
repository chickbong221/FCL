#!/bin/bash
# Sweep for: stgm_gamma

python system/main.py --cfp ./hparams/sweep_STGM/stgm_gamma/stgm_gamma_0.1.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/stgm_gamma/stgm_gamma_0.32.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/stgm_gamma/stgm_gamma_0.55.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/stgm_gamma/stgm_gamma_0.78.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/stgm_gamma/stgm_gamma_1.0.json --wandb True