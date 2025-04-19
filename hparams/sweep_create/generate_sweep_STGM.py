import os
import json
import numpy as np
from copy import deepcopy

base_config = {
    "optimizer": "sgd",
    "datadir": "dataset",
    "device": "cuda",
    "device_id": "0",
    "dataset": "IMAGENET1k",
    "num_classes": 1000,
    "model": "CNN",
    "batch_size": 64,
    "local_learning_rate": 0.005,
    "learning_rate_decay": False,
    "learning_rate_decay_gamma": 0.99,
    "global_rounds": 100,
    "local_epochs": 1,
    "algorithm": "FedSTGM",
    "join_ratio": 1.0,
    "random_join_ratio": False,
    "num_clients": 10,
    "prev": 0,
    "times": 1,
    "eval_gap": 1,
    "out_folder": "out",
    "note": None,
    "num_tasks": 500,
    "client_drop_rate": 0.0,
    "time_threthold": 10000,
    "stgm_rounds": 100,
    "stgm_learning_rate": 25,
    "stgm_momentum": 0.5,
    "stgm_step_size": 30,
    "stgm_gamma": 0.5,
    "stgm_c": 0.25,
    "stgm_meta_lr": 0.5,
    "grad_balance": False,
    "coreset": True,
    "tgm": True,
    "sgm": True
}

# Sweep parameters (already fixed to Python types)
sweep_params = {
    "stgm_rounds": [int(x) for x in np.linspace(20, 100, 5)],
    "stgm_learning_rate": [int(x) for x in np.linspace(5, 50, 5)],
    "stgm_momentum": [float(np.round(x, 2)) for x in np.linspace(0.1, 1.0, 5)],
    "stgm_step_size": [int(x) for x in np.linspace(10, 50, 5)],
    "stgm_gamma": [float(np.round(x, 2)) for x in np.linspace(0.1, 1.0, 5)],
    "stgm_c": [float(np.round(x, 2)) for x in np.linspace(0.1, 1.0, 5)],
    "stgm_meta_lr": [float(np.round(x, 2)) for x in np.linspace(0.1, 1.0, 5)],
    "grad_balance": [False, True],
    "local_epochs": [int(x) for x in np.arange(1, 11)],
    "local_learning_rate": [float(np.round(x, 5)) for x in np.linspace(0.001, 0.05, 5)],
}

# Root folder
os.makedirs("../sweep_STGM", exist_ok=True)

# Loop through each param to sweep
for param_name, values in sweep_params.items():
    folder = os.path.join("../sweep_STGM", param_name)
    os.makedirs(folder, exist_ok=True)

    for i, value in enumerate(values):
        config = deepcopy(base_config)
        config[param_name] = value
        config["note"] = f"{param_name}={value}"

        filename = os.path.join(folder, f"{param_name}_{value}.json")
        with open(filename, "w") as f:
            json.dump(config, f, indent=4)
