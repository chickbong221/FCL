import os
import sys
import copy
import torch
import argparse
import time
import warnings
import numpy as np
import torchvision
import logging
import wandb

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverala import FedALA
from flcore.servers.serverdbe import FedDBE
from flcore.servers.serverstgm import FedSTGM
from flcore.servers.serverfcil import FedFCIL

from flcore.trainmodel.models import *

from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)


def run(args):
    if args.wandb:
        wandb.login(key="b1d6eed8871c7668a889ae74a621b5dbd2f3b070")
        wandb.init(
            project="FCL",
            entity="letuanhf-hanoi-university-of-science-and-technology",
            config=args, 
            name=f"{args.dataset}_{args.model}_localep{args.local_epochs}",
        )

    time_list = []
    reporter = MemReporter()
    model_str = args.model

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
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif model_str == "ResNet18":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
        elif model_str == "ResNet10":
            args.model = resnet10(num_classes=args.num_classes).to(args.device)
        else:
            raise NotImplementedError

        print(args.model)

        # select algorithm
        if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)

        elif args.algorithm == "FedALA":
            server = FedALA(args, i)

        elif args.algorithm == "FedDBE":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedDBE(args, i)
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
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)
    print("All done!")
    reporter.report()

if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--datadir", type=str, default="dataset")
    parser.add_argument("--data_split_file", type=str, default="data_split/CIFAR100_split.pkl")
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="CIFAR100", choices=['EMNIST-Letters', 'EMNIST-Letters-malicious', 
                                                                            'EMNIST-Letters-shuffle', 'CIFAR100', 'MNIST-SVHN-FASHION', 'IMAGENET1k'])
    parser.add_argument('-ncl', "--num_classes", type=int, default=100)
    parser.add_argument('-m', "--model", type=str, default="CNN")
    parser.add_argument('-lbs', "--batch_size", type=int, default=64)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100, 
                        help="For auto_break")
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=10,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=32000, 
                        help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.")
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-graind than its original paper.")
    # FedDBE
    parser.add_argument('-mo', "--momentum", type=float, default=0.1)
    parser.add_argument('-klw', "--kl_weight", type=float, default=0.0)

    # FedSTGM
    parser.add_argument('-mem', "--stgm_mem", type=int, default=1e7)
    parser.add_argument('-car', "--grad_stgm_rounds", type=int, default=100)
    parser.add_argument('-calr', "--grad_stgm_learning_rate", type=float, default=25)
    parser.add_argument('-mmt', "--stgm_momentum", type=float, default=0.5)
    parser.add_argument('-ss', "--step_size", type=int, default=30)
    parser.add_argument('-gam', "--gamma", type=float, default=0.5)
    parser.add_argument('-c', "--c_parameter", type=float, default=0.5)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    for arg in vars(args):
        print(arg, '=',getattr(args, arg))
    print("=" * 50)
    run(args)

