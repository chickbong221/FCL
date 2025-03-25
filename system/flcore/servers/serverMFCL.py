import time
import torch
import copy
import glog as logger
from flcore.clients.clientMFCL import clientMFCL
from flcore.servers.serverbase import Server
from flcore.trainmodel.fedMFCL_models import *
from flcore.utils.fedMFCL_utils import *
from utils.model_utils import read_client_data_FCL, read_client_data_FCL_imagenet1k
import numpy as np

class FedMFCL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.Budget = []
        self.set_slow_clients()
        self.set_clients(clientMFCL)

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            
            if self.args.dataset == 'IMAGENET1k':
                id, train_data, test_data, label_info = read_client_data_FCL_imagenet1k(i, task=0, classes_per_task=2, count_labels=True)
            else:
                id, train_data, test_data, label_info = read_client_data_FCL(i, self.data, dataset=self.args.dataset, count_labels=True, task=0)

            # count total samples (accumulative)
            self.total_train_samples +=len(train_data)
            self.total_test_samples += len(test_data)
            id = i

            client = clientObj(self.args, 
                            id=i, 
                            train_data=train_data,
                            test_data=test_data,
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow,
                            kd_weight=self.args.w_kd,
                            ft_weight=self.args.w_ft,
                            syn_size=self.args.syn_size)
            self.clients.append(client)

            # update classes so far & current labels
            client.classes_so_far.extend(label_info['labels'])
            client.current_labels.extend(label_info['labels'])
            client.task_dict[0] = label_info['labels']

        logger.info("Number of Train/Test samples: %d/%d"%(self.total_train_samples, self.total_test_samples))
        logger.info("Finished creating FedAvg server.")


    def train(self):
        # Initialization
        teacher, generator = None, None
        gamma = np.log(self.args.lr_end / self.args.lr)
        classes_learned = 0 
        generator = GeneratorBig(zdim=self.args.z_dim, in_channel=3, img_sz=32, convdim=self.args.conv_dim)

        # Task
        if self.args.dataset == 'IMAGENET1k':
            N_TASKS = 2
        else:
            N_TASKS = len(self.data['train_data'][self.data['client_names'][0]]['x'])
        print(str(N_TASKS) + " tasks are available")

        for task in range(N_TASKS):

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
                        id, train_data, test_data, label_info = read_client_data_FCL_imagenet1k(i, task=task, classes_per_task=2, count_labels=True)
                    else:
                        id, train_data, test_data, label_info = read_client_data_FCL(i, self.data, dataset=self.args.dataset, count_labels=True, task=task)
                    
                    self.clients[i].next_task(train_data, test_data, label_info)

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

            for i in range(self.global_rounds):

                lr = self.args.lr * np.exp(i / self.args.global_rounds * gamma)
                glob_iter = i + self.global_rounds * task
                self.updates = [] 
                s_t = time.time()

                self.selected_clients = self.select_clients()
                # self.global_model =  network(numclass=self.args.num_classes_per_task, feature_extractor=self.global_model)
                self.send_models()

                if i%self.eval_gap == 0:
                    print(f"\n-------------Round number: {i}-------------")
                    print("\nEvaluate global model")
                    self.evaluate(glob_iter=glob_iter)

                for client in self.selected_clients:
                    model = copy.deepcopy(self.global_model)
                    client.train(model, lr, teacher, generator)
                    self.updates.append(model.state_dict())

                aggr = fedavg_aggregation(self.updates)
                self.set_weights(aggr)

                self.Budget.append(time.time() - s_t)
                print('-'*25, 'time cost', '-'*25, self.Budget[-1])

                if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                    break

            print("\nBest accuracy.")
            print(max(self.rs_test_acc))
            print("\nAverage time cost per round.")
            print(sum(self.Budget[1:])/len(self.Budget[1:]))

            if task == N_TASKS - 1:
                original_global = deepcopy(self.global_model)
                teacher = train_gen(deepcopy(self.global_model), classes_learned, generator, self.args)
                for client in self.clients:
                    client.last_valid_dim = classes_learned
                    client.valid_dim = classes_learned + self.args.num_classes_per_task
                self.global_model = original_global  
                classes_learned += self.args.num_classes_per_task
                self.global_model.Incremental_learning(classes_learned)

            # self.save_results()
            # self.save_global_model()

            if self.num_new_clients > 0:
                self.eval_new_clients = True
                self.set_new_clients(clientMFCL)
                print(f"\n-------------Fine tuning round-------------")
                print("\nEvaluate new clients")
                self.evaluate(glob_iter=glob_iter)

    def set_weights(self, weights):
        self.global_weights = weights