import glog as logger
import torch
import time
import numpy as np

from flcore.clients.clientprecise import ClientPreciseFCL
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k

class FedPrecise(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.classifier_head_list = ['classifier.fc_classifier', 'classifier.fc2']
        
        # select slow clients
        self.set_slow_clients()
        self.set_clients(ClientPreciseFCL)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
    
    def train(self):
        
        for task in range(self.N_TASKS):

            print(f"\n================ Current Task: {task} =================")
            if task == 0:
                 # update labels info. for the first task
                available_labels = set()
                available_labels_current = set()
                available_labels_past = set()
                for u in self.clients:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(set(u.current_labels))

                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)

            else:
                self.current_task = task
                
                torch.cuda.empty_cache()
                for i in range(len(self.clients)):
                    
                    if self.args.dataset == 'IMAGENET1k':
                        train_data, test_data, label_info = read_client_data_FCL_imagenet1k(i, task=task, classes_per_task=2, count_labels=True)
                    elif self.args.dataset == 'CIFAR100':
                        train_data, test_data, label_info = read_client_data_FCL_cifar100(i, task=task, classes_per_task=2, count_labels=True)
                    else:
                        raise NotImplementedError("Not supported dataset")

                    # update dataset
                    self.clients[i].next_task(train_data, test_data, label_info) # assign dataloader for new data
                    # print(self.clients[i].task_dict)

                # update labels info.
                available_labels = set()
                available_labels_current = set()
                available_labels_past = self.clients[0].available_labels
                for u in self.clients:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(set(u.current_labels))

                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)
            
            # ============ train ==============

            for i in range(self.global_rounds):
                
                glob_iter = i + self.global_rounds * task
                s_t = time.time()

                self.selected_clients = self.select_clients()
                self.send_models()

                if i%self.eval_gap == 0:
                    print(f"\n-------------Round number: {i}-------------")
                    self.eval(task=task, glob_iter=glob_iter, flag="global")

                global_classifier = self.global_model.classifier
                global_classifier.eval()
                
                for client in self.selected_clients:
                    verbose = False
                    client.train(glob_iter, global_classifier, verbose=verbose)

                self.receive_models()
                self.aggregate_parameters()

                if i%self.eval_gap == 0:
                    self.eval(task=task, glob_iter=glob_iter, flag="local")

                self.Budget.append(time.time() - s_t)
                print('-'*25, 'time cost', '-'*25, self.Budget[-1])
                    
            self.eval_task(task=task, glob_iter=glob_iter, flag="local")
            
            # need eval before data update
            self.send_models()
            self.eval_task(task=task, glob_iter=glob_iter, flag="global")

    def aggregate_parameters(self, class_partial=False):
        assert (self.selected_clients is not None and len(self.selected_clients) > 0)
        
        param_dict = {}
        for name, param in self.global_model.named_parameters():
            param_dict[name] = torch.zeros_like(param.data)
        
        total_train = 0
        for client in self.selected_clients:
            total_train += len(client.train_data) # length of the train data for weighted importance
        
        param_weight_sum = {}
        for client in self.selected_clients:
            for name, param in client.model.named_parameters():
                if ('fc_classifier' in name and class_partial):
                    class_available = torch.Tensor(client.classes_so_far).long()
                    param_dict[name][class_available] += param.data[class_available] * len(client.train_data) / total_train
                    
                    add_weight = torch.zeros([param.data.shape[0]]).cuda()
                    add_weight[class_available] = len(client.train_data) / total_train
                else:
                    param_dict[name] += param.data * len(client.train_data) / total_train
                    add_weight = len(client.train_data) / total_train
                
                if name not in param_weight_sum.keys():
                    param_weight_sum[name] = add_weight
                else:
                    param_weight_sum[name] += add_weight
                
        for name, param in self.global_model.named_parameters():

            if 'fc_classifier' in name and class_partial:
                valid_class = (param_weight_sum[name]>0)
                weight_sum = param_weight_sum[name][valid_class]
                if 'weight' in name:
                    weight_sum = weight_sum.view(-1, 1)
                param.data[valid_class] = param_dict[name][valid_class]/weight_sum
            else:
                param.data = param_dict[name]/param_weight_sum[name]

    def add_parameters(self, client, ratio, partial=False):
        if partial:
            for server_param, client_param in zip(self.global_model.get_shared_parameters(), client.model.get_shared_parameters()):
                server_param.data = server_param.data + client_param.data.clone() * ratio
        else:
            # replace all!
            for server_param, client_param in zip(self.global_model.parameters(), client.model.parameters()):
                server_param.data = server_param.data + client_param.data.clone() * ratio

    def set_clients(self, clientObj):
        total_clients = 10
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            
            if self.args.dataset == 'IMAGENET1k':
                train_data, test_data, label_info = read_client_data_FCL_imagenet1k(i, task=0, classes_per_task=2, count_labels=True)
            elif self.args.dataset == 'CIFAR100':
                train_data, test_data, label_info = read_client_data_FCL_cifar100(i, task=0, classes_per_task=2, count_labels=True)
            else:
                raise NotImplementedError("Not supported dataset")

            client = clientObj(self.args, 
                        id=i,
                        train_data=train_data,
                        test_data=test_data,
                        classifier_head_list = self.classifier_head_list,
                        train_slow=train_slow, 
                        send_slow=send_slow)

            self.clients.append(client)

            # update classes so far & current labels
            client.classes_so_far.extend(label_info['labels'])
            client.current_labels.extend(label_info['labels'])
            client.task_dict[0] = label_info['labels']