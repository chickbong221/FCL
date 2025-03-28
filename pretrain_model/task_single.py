import os
import sys
import torch.nn as nn
import torch
import argparse
import time
import warnings
import numpy as np
import torchvision
import wandb
from torch.utils.data import DataLoader

import sys
sys.path.append("/media/tuannl1/heavy_weight/FCL/PFLlib/system")

from flcore.trainmodel.models import *

from flcore.trainmodel.precise_models import PreciseModel
from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *

from pretrain_utils import read_all_client_data_FCL_imagenet1k_task_single, read_all_client_data_FCL_cifar100_task_single

warnings.simplefilter("ignore")
torch.manual_seed(0)

# Define the dataset-specific parameters
def get_num_tasks(dataset):
    if dataset == "IMAGENET1k":
        return 500
    elif dataset == "CIFAR100":
        return 50
    else:
        raise NotImplementedError("Not supported dataset")

# Define model selection
def get_model(model_name, num_classes, device):
    if model_name == "CNN":
        return FedAvgCNN(in_features=3, num_classes=num_classes, dim=1600).to(device)
    elif model_name == "LeNet_big":
        return LeNet_big(in_features=3, num_classes=num_classes, dim=4096).to(device)
    elif model_name == "LeNet_normal":
        return LeNet_normal(in_features=3, num_classes=num_classes, dim=1600).to(device)
    elif model_name == "ResNet18":
        return torchvision.models.resnet18(pretrained=False, num_classes=num_classes).to(device)
    elif model_name == "ResNet10":
        return resnet10(num_classes=num_classes).to(device)
    elif model_name == "PreciseModel":
        return PreciseModel(num_classes=num_classes).to(device)
    else:
        raise NotImplementedError(f"Unsupported model: {model_name}")

# Define optimizer selection
def get_optimizer(optimizer_name, model, learning_rate):
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

# Training function
def train_and_evaluate(args, task, model, trainloader, testloader, optimizer, num_epochs, device):
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(trainloader)
        
        # Evaluate model
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                _, predicted = torch.max(output, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
        
        acc = 100 * correct / total

        if args.wandb:
            wandb.log({
                "Global/Averaged Train Loss": avg_loss,
                "Global/Averaged Test Accurancy": acc,
            }, step=epoch+1)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}, Test Accuracy: {acc:.2f}%")

    torch.save(model.state_dict(), f"task_single_model/{task}.pth")

# Main execution function
def run():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--dataset", type=str, default="IMAGENET1k", choices=["CIFAR100", "IMAGENET1k"])
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--model", type=str, default="CNN")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--device_id", type=str, default="0")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    
    num_tasks = get_num_tasks(args.dataset)
    print(f"{num_tasks} tasks are available")
    
    for task in range(num_tasks):

        # Setup Wandb
        if args.wandb:
            wandb.login(key="b1d6eed8871c7668a889ae74a621b5dbd2f3b070")
            wandb.init(
                project="FCL-pretrained",
                entity="letuanhf-hanoi-university-of-science-and-technology",
                config=args, 
                name=f"{args.dataset}_{task}" 
            )

        model = get_model(args.model, args.num_classes, device)
        optimizer = get_optimizer(args.optimizer, model, args.learning_rate)
        
        # Load dataset
        if args.dataset == "CIFAR100":
            train_data, test_data, _ = read_all_client_data_FCL_cifar100_task_single(args.num_clients, task, 2, True)
        elif args.dataset == "IMAGENET1k":
            train_data, test_data, _ = read_all_client_data_FCL_imagenet1k_task_single(args.num_clients, task, 2, True)
        
        trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
        testloader = DataLoader(test_data, batch_size=args.batch_size, drop_last=True)
        
        train_and_evaluate(args, task, model, trainloader, testloader, optimizer, args.epochs, device)

if __name__ == "__main__":
    total_start = time.time()
    run()