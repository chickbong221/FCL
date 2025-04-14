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
    # general
    parser.add_argument("--offlog", type=bool, default=False)
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--datadir", type=str, default="dataset")
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
    parser.add_argument('-sfn', "--out_folder", type=str, default='out')
    parser.add_argument("--note", type=str, default=None)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-grained than its original paper.")
    # FedDBE
    parser.add_argument('-mo', "--momentum", type=float, default=0.1)
    parser.add_argument('-klw', "--kl_weight", type=float, default=0.0)

    #FedWeIT
    parser.add_argument('--output_path', type=str, default='output_fedweit/', help="Path to save outputs")
    parser.add_argument('--sparse_comm', type=bool, default=True, help="Enable sparse communication")
    parser.add_argument('--client_sparsity', type=float, default=0.3, help="Client-side sparsity level")
    parser.add_argument('--server_sparsity', type=float, default=0.3, help="Server-side sparsity level")
    parser.add_argument('--base_network', type=str, default='lenet', choices=['lenet'], help="Base network architecture")
    parser.add_argument('--lr_patience', type=int, default=3, help="Patience for learning rate scheduling")
    parser.add_argument('--lr_factor', type=int, default=3, help="Factor for learning rate reduction")
    parser.add_argument('--lr_min', type=float, default=1e-10, help="Minimum learning rate")
    parser.add_argument('--wd', type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--lambda_l1', type=float, default=1e-3, help="L1 regularization coefficient")
    parser.add_argument('--lambda_l2', type=float, default=100.0, help="L2 regularization coefficient")
    parser.add_argument('--lambda_mask', type=float, default=0.0, help="Mask regularization coefficient")
    parser.add_argument('--num_tasks', type=int, default=500, help="num tasks")

    # PreciseFCL
    parser.add_argument("--k_loss_flow", type=float, default=0.1)
    parser.add_argument("--k_kd_global_cls", type=float, default=0)
    parser.add_argument("--k_kd_last_cls", type=float, default=0.2)
    parser.add_argument("--k_kd_feature", type=float, default=0.5)
    parser.add_argument("--k_kd_output", type=float, default=0.1)
    parser.add_argument("--k_flow_lastflow", type=float, default=0.4)
    parser.add_argument("--flow_epoch", type=int, default=5)
    parser.add_argument("--flow_explore_theta", type=float, default=0.2)
    parser.add_argument("--classifier_global_mode", type=str, default='all', help='[head, extractor, none, all]')
    parser.add_argument('--flow_lr', type=float, default=1e-4)  
    parser.add_argument('--fedprox_k', type=float, default=0) 
    parser.add_argument('--use_lastflow_x', action="store_true")
    parser.add_argument("--beta", type=float, default=1.0, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg") 
    parser.add_argument('--c_channel_size', type=int, default=64)

    args = parser.parse_args()
    args.log_dir = os.path.join(args.output_path, 'logs/{}-{}'.format(args.model, args.dataset))
    args.state_dir = os.path.join(args.output_path, 'states/{}-{}'.format(args.model, args.dataset))
    
    # FedSTGM
    parser.add_argument('-car', "--grad_stgm_rounds", type=int, default=100)
    parser.add_argument('-calr', "--grad_stgm_learning_rate", type=float, default=25)
    parser.add_argument('-mmt', "--stgm_momentum", type=float, default=0.5)
    parser.add_argument('-ss', "--step_size", type=int, default=30)
    parser.add_argument('-gam', "--gamma", type=float, default=0.5)
    parser.add_argument('-c', "--c_parameter", type=float, default=0.5)

    #FCIL
    parser.add_argument('-mem', "--memory_size", type=int, default=2000)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    run(args)

