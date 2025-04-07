import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k

class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_data, test_data, **kwargs):
        torch.manual_seed(0)
        self.model = copy.deepcopy(args.model)
        self.args = args
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer

        self.num_classes = args.num_classes
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        self.train_source = [image for image, _ in self.train_data]
        self.train_targets = [label for _, label in self.train_data]

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()

        if args.algorithm != "PreciseFCL":
            if args.optimizer == "sgd":
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
            elif args.optimizer == "adam":
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            else:
                raise ValueError(f"Unsupported optimizer: {args.optimizer}.")
            self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer, 
                gamma=args.learning_rate_decay_gamma
            )
            self.learning_rate_decay = args.learning_rate_decay

        # continual federated learning
        if self.args.dataset == 'IMAGENET1k':
            self.N_TASKS = 500
        elif self.args.dataset == 'CIFAR100':
            self.N_TASKS = 50
        else:
            raise NotImplementedError("Not supported dataset")
        print("Anh Duong dep trai")
        
        self.test_data_all_task = []
        for task in range(self.N_TASKS):
            
            if self.args.dataset == 'IMAGENET1k':
                _, test_data, _ = read_client_data_FCL_imagenet1k(self.id, task=task, classes_per_task=2, count_labels=True)
            elif self.args.dataset == 'CIFAR100':
                _, test_data, _ = read_client_data_FCL_cifar100(self.id, task=task, classes_per_task=2, count_labels=True)
            else:
                raise NotImplementedError("Not supported dataset")

            self.test_data_all_task.append(test_data)

        self.classes_so_far = [] # all labels of a client so far 
        self.available_labels_current = [] # labels from all clients on T (current)
        self.current_labels = [] # current labels for itself
        self.classes_past_task = [] # classes_so_far (current labels excluded) 
        self.available_labels_past = [] # labels from all clients on T-1
        self.available_labels = [] # l from all c from 0-T
        self.current_task = 0
        self.task_dict = {}
        self.last_copy = None
        self.if_last_copy = False
        self.args = args

    def next_task(self, train, test, label_info = None, if_label = True):
        
        if self.args.algorithm != "PreciseFCL" and self.learning_rate_decay:
            # update last model:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate  # Đặt lại về giá trị ban đầu

            self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer, 
                gamma=self.args.learning_rate_decay_gamma
            )

        self.last_copy  = copy.deepcopy(self.model)
        self.last_copy.cuda()
        self.if_last_copy = True
        
        # update dataset: 
        self.train_data = train
        self.test_data = test

        self.train_targets = [label for _, label in self.train_data]
        
        # update classes_past_task
        self.classes_past_task = copy.deepcopy(self.classes_so_far)
        
        # update class recorder:
        self.current_task += 1

        # update classes_so_far
        if if_label:
            self.classes_so_far.extend(label_info['labels'])
            self.task_dict[self.current_task] = label_info['labels']

            self.current_labels.clear()
            self.current_labels.extend(label_info['labels'])
            
        return

    def assign_task_id(self, task_dict):
        if not isinstance(task_dict, dict):
            raise ValueError("task_dict must be a dictionary")

        label_key = tuple(sorted(self.current_labels)) if isinstance(self.current_labels,
                                                                     (set, list)) else self.current_labels

        return task_dict.get(label_key, -1)  # Returns -1 if labels are not in task_dict

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = self.train_data
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, task, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = self.test_data_all_task[task]
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self, task):
        testloader = self.load_test_data(task=task)
        # self.model = self.load_model('model')
        # self.model.to(self.device)

        self.model.eval()

        test_acc = 0
        test_num = 0
        
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
        
        return test_acc, test_num

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                loss = self.loss(output, y)
                
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
