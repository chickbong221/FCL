import os
import sys
import copy
import torch
import argparse
import time
import warnings
import numpy as np
import torchvision
import json
import wandb
from argparse import Namespace
from types import SimpleNamespace

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverala import FedALA
from flcore.servers.serverdbe import FedDBE
from flcore.servers.serveras import FedAS
from flcore.servers.serverweit import FedWeIT
from flcore.servers.serverprecise import FedPrecise

from flcore.trainmodel.models import *

from flcore.trainmodel.precise_models import PreciseModel
from flcore.servers.serverstgm import FedSTGM
from flcore.servers.serverfcil import FedFCIL

from flcore.trainmodel.bilstm import *
# from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *

warnings.simplefilter("ignore")
torch.manual_seed(0)


def run(args):
    
    if args.wandb:
        wandb.login(key="b1d6eed8871c7668a889ae74a621b5dbd2f3b070")
        wandb.init(
            project="FCL",
            entity="letuanhf-hanoi-university-of-science-and-technology",
            config=args, 
            name=f"{args.dataset}_{args.model}_{args.algorithm}_{args.optimizer}_lr{args.local_learning_rate}_{args.note}" if args.note else f"{args.dataset}_{args.model}_{args.algorithm}_{args.optimizer}_lr{args.local_learning_rate}", 
        )

    time_list = []
    model_str = args.model
    args.model_str = model_str

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "CNN": # non-convex
            if "CIFAR100" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "IMAGENET1k" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "IMAGENET1k224" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=179776).to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif model_str == "LeNet_big":
            args.model = LeNet_big(in_features=3, num_classes=args.num_classes, dim=4096).to(args.device)
        elif model_str == "LeNet_normal":
            args.model = LeNet_normal(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
        elif model_str == "ResNet50":
            args.model = torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes).to(args.device)
        elif model_str == "ResNet50-pretrained":
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
            args.model = torchvision.models.resnet50(weights=weights, num_classes=args.num_classes).to(args.device)
        elif model_str == "ResNet18":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
        elif model_str == "Swin_t":
            args.model = torchvision.models.swin_t(weights=None, num_classes=args.num_classes).to(args.device)
        elif model_str == "PreciseModel":    
            args.model = PreciseModel(args).to(args.device)
        else:
            raise NotImplementedError

        # select algorithm
        if args.algorithm == "FedAvg":
            server = FedAvg(args, i)

        elif args.algorithm == "FedALA":
            server = FedALA(args, i)

        elif args.algorithm == "FedDBE":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedDBE(args, i)

        elif args.algorithm == "FedWeIT":
            server = FedWeIT(args, i)

        elif args.algorithm == "PreciseFCL":
            server = FedPrecise(args, i)

        elif args.algorithm == 'FedAS':

            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAS(args, i)

        elif args.algorithm == "FedFCIL":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedFCIL(args, i)
            
        elif args.algorithm == "FedSTGM":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedSTGM(args, i)

        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    
    # Global average
    print("All done!")

if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfp', type=str, default="./hparams/FedAvg.json", help='Configuration path for training')
    parser.add_argument('--note', type=str, default=None, help='Optional note to add to save name')
    parser.add_argument('--wandb', type=bool, default=False, help='Log on wandb')
    parser.add_argument('--offlog', type=bool, default=False, help='Save wandb logger')
    parser.add_argument('--log', type=bool, default=False, help='Print logger')

    args = parser.parse_args()

    with open(args.cfp, 'r') as f:
        cfdct = json.load(f)
    if args.note is not None:
        cfdct['note'] = args.note

    cfdct['wandb'] = args.wandb
    cfdct['offlog'] = args.offlog
    cfdct['log'] = args.log

    args = Namespace(**cfdct)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    run(args)

