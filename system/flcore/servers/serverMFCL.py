import time
import torch
import glog as logger
from flcore.clients.clientMFCL import clientMFCL
from flcore.servers.serverbase import Server
from utils.model_utils import read_client_data_FCL, read_client_data_FCL_imagenet1k
import numpy as np

class FedMFCL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.Budget = []
        self.set_slow_clients()
        self.set_clients(clientMFCL)


    def set_clients(self, clientObj):
        total_clients = 10
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
                            initial_weights=self.global_weights)
            self.clients.append(client)
            
            # update classes so far & current labels
            client.classes_so_far.extend(label_info['labels'])
            client.current_labels.extend(label_info['labels'])

        logger.info("Number of Train/Test samples: %d/%d"%(self.total_train_samples, self.total_test_samples))
        logger.info("Data from {} clients in total.".format(total_clients))
        logger.info("Finished creating FedMFCL server.")


    def train(self):
        teacher, generator = None, None
        gamma = np.log(self.args.lr_end / self.args.lr)
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

                    # update dataset
                    self.clients[i].train_data = train_data
                    self.clients[i].test_data = test_data
                    self.clients[i].train_samples = len(train_data)
                    self.clients[i].test_samples = len(test_data)
                    self.clients[i].initial_weights = self.global_weights
                    self.clients[i].classes_so_far.extend(label_info['labels'])
                    self.clients[i].current_labels.extend(label_info['labels'])

                    # update available labels
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
                
                print("Change glob round")
                lr = self.args.lr * np.exp(i / self.args.global_rounds * gamma)
                glob_iter = i + self.global_rounds * task
                self.updates = []
                self.curr_round = glob_iter+1
                self.is_last_round = i==0
                if self.is_last_round:
                    self.client_adapts = []
                s_t = time.time()
                self.selected_clients = self.select_clients()
                self.send_models()

                if i%self.eval_gap == 0:
                    print(f"\n-------------Round number: {i}-------------")
                    print("\nEvaluate global model")
                    self.evaluate(glob_iter=glob_iter)

                for client in self.selected_clients:
                    update = client.train(self.get_weights, lr, teacher, generator)
                    if not update == None:
                        self.updates.append(update)

                aggr = self.train.aggregate(self.updates)
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
                pass

            # self.save_results()
            # self.save_global_model()

            if self.num_new_clients > 0:
                self.eval_new_clients = True
                self.set_new_clients(clientMFCL)
                print(f"\n-------------Fine tuning round-------------")
                print("\nEvaluate new clients")
                self.evaluate(glob_iter=glob_iter)

    def get_weights(self):
        return self.global_weights

    def set_weights(self, weights):
        self.global_weights = weights