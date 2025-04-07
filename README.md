# Federated Continual Learning Benchmark

## Overview
Federated Continual Learning (FCL) Benchmark is a standardized evaluation framework for assessing continual learning methods in federated settings. It provides datasets, evaluation metrics, and baseline implementations to facilitate research in FCL.

## Features
- **Diverse Datasets**: Supports multiple datasets commonly used in CFL research.
- **Baseline Models**: Includes various baseline models for comparison.
- **Customizable**: Easily extendable for new datasets and algorithms.
- **Federated Learning Simulation**: Implements a federated learning environment for continual learning.
- **Metrics & Logging**: Provides standardized metrics for evaluating performance over time.

## Installation
```sh
# Clone the repository
git clone git@github.com:chickbong221/FCL.git
cd FCL

# Create a virtual environment (optional but recommended)
python -m venv .env
source .env/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download and preprocess data
python dataset/cifar100_npy.py
```

## Usage
### Running an Experiment
```sh
python system/main.py --dataset IMAGENET1k --num_classes 1000 --wandb True 
python system/main.py --dataset CIFAR100 --num_classes 100 -algo PreciseFCL -m PreciseModel -gr 1000 --wandb True -did 0 -lr 1e-4 --flow_lr 1e-4 --optimizer adam
python system/main.py --dataset CIFAR100 --num_classes 100  -did 0 -gr 1000
python system/main.py --dataset CIFAR100 --num_classes 100  -did 0 -gr 1000 -algo FedDBE
python system/main.py --dataset CIFAR100 --num_classes 100  -did 0 -gr 1000 -algo FedALA
python system/main.py --dataset CIFAR100 --num_classes 100  -did 0 -gr 1000 -algo FedAS
```

## Benchmarked Algorithms
- **FedAvg** (Federated Averaging)
- **PreciseFCL** 
- **FedWeIT** 
- **FedALA** 
- **FedAS**
- **FedDBE**

## Datasets
- PMNIST
- CIFAR-100
- IMAGENET1k

## Contributing
We welcome contributions! Please check our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact
For questions or collaborations, please open an issue or reach out to `Anh-Duong-dep-trai@example.com`.
