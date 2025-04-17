import time
import torch
# from flcore.clients.clientMFCL import clientMFCL
from flcore.clients.clientMFCL import clientMFCL
from flcore.servers.serverbase import Server
from threading import Thread
from utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k
import numpy as np
from flcore.utils_core.FedMFCL_utils import *
from flcore.trainmodel.FedMFCL_model import FedMFCL_Network
from copy import deepcopy



class FedMFCL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientMFCL)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def set_clients(self, clientObj):

        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            
            if self.args.dataset == 'IMAGENET1k':
                train_data, test_data, label_info = read_client_data_FCL_imagenet1k(i, task=0, classes_per_task=2, count_labels=True)
            elif self.args.dataset == 'CIFAR100':
                train_data, test_data, label_info = read_client_data_FCL_cifar100(i, task=0, classes_per_task=2, count_labels=True)
            else:
                raise NotImplementedError("Not supported dataset")

            # count total samples (accumulative)
            # self.total_train_samples +=len(train_data)
            # self.total_test_samples += len(test_data)
            # id = i

            client = clientObj(self.args, 
                            id=i, # client id
                            train_data=train_data,
                            test_data=test_data,
                            train_slow=train_slow, 
                            send_slow=send_slow,
                            kd_weight=self.args.w_kd,
                            ft_weight=self.args.w_ft,
                            syn_size=self.args.syn_size)
            self.clients.append(client)

            # update imagenet1k-classes so far & current labels
            client.classes_so_far.extend(label_info['labels'])
            client.current_labels.extend(label_info['labels'])
            client.task_dict[0] = label_info['labels']

    def train(self):
        
        teacher, generator = None, None
        gamma = np.log(self.args.lr_end / self.args.lr)
        task_size, classes_learned = 2, 2

        if self.args.dataset == 'IMAGENET1k':
            generator = GeneratorBig(zdim=self.args.z_dim, in_channel=3, img_sz=32, convdim=self.args.conv_dim)
        elif self.args.dataset == 'CIFAR100':
            generator = Generator(zdim=self.args.z_dim, in_channel=3, img_sz=32, out_channel=128)  


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

                """
                update for task 0
                """

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
                """Label from client 0"""

                for u in self.clients:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(set(u.current_labels))

                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)

                    # print(available_labels)

            # ============ train ==============

            for i in range(self.global_rounds):

                lr = self.args.lr * np.exp(i / self.args.global_rounds * gamma)
                glob_iter = i + self.global_rounds * task
                self.update = []
                s_t = time.time()

                self.selected_clients = self.select_clients()
                self.global_model = FedMFCL_Network(feature_extractor=self.args.model,num_classes=100)
                self.send_models()

                if i%self.eval_gap == 0:
                    print(f"\n-------------Round number: {i}-------------")
                    self.eval(task=task, glob_iter=glob_iter, flag="global")

                for client in self.selected_clients:
                    model = deepcopy(self.global_model)
                    client.train(model, lr, teacher, generator)

                # aggr = fedavg_aggregation(self.updates)
                # self.set_weights(aggr)
                self.receive_models()
                self.aggregate_parameters()

                if i%self.eval_gap == 0:
                    self.eval(task=task, glob_iter=glob_iter, flag="local")

                self.Budget.append(time.time() - s_t)
                # print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            self.eval_task(task=task, glob_iter=glob_iter, flag="local")
            
            # need eval before data update
            self.send_models()
            self.eval_task(task=task, glob_iter=glob_iter, flag="global")

            if task != self.N_TASKS - 1:
                original_global = deepcopy(self.global_model)
                teacher = train_gen(deepcopy(self.global_model), classes_learned, generator, self.args)
                for client in self.clients:
                    client.last_valid_dim = classes_learned
                    # client.valid_dim = classes_learned + task_size
                    client.valid_dim = 100
                self.global_model = original_global  
                classes_learned += task_size
                # self.global_model.Incremental_learning(classes_learned)

"""
update FedMFCL_utils
update FedMFCL_server
update FedMFCL_model
update send_model
update receive_model

"""