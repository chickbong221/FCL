#!/bin/bash
# Sweep for: local_learning_rate

python system/main.py --cfp ./hparams/sweep_STGM/local_learning_rate/local_learning_rate_0.001.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/local_learning_rate/local_learning_rate_0.01325.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/local_learning_rate/local_learning_rate_0.0255.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/local_learning_rate/local_learning_rate_0.03775.json --wandb True
python system/main.py --cfp ./hparams/sweep_STGM/local_learning_rate/local_learning_rate_0.05.json --wandb True