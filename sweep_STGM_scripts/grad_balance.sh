#!/bin/bash
# Sweep for: grad_balance

python system/main.py --cfp ./hparams/sweep_STGM/grad_balance/grad_balance_False.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/grad_balance/grad_balance_True.json --wandb True