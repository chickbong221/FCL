import time
import torch
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
from utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
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

                    # print(available_labels)

            # ============ train ==============

            for i in range(self.global_rounds):

                glob_iter = i + self.global_rounds * task
                s_t = time.time()

                self.evaluate_before_sendmodel(task=task, glob_iter=glob_iter)

                self.selected_clients = self.select_clients()
                self.send_models()

                if i%self.eval_gap == 0:
                    # print(f"\n-------------Round number: {i}-------------")
                    # print("\nEvaluate global model")
                    self.evaluate(task=task, glob_iter=glob_iter)

                for client in self.selected_clients:
                    client.train()

                # threads = [Thread(target=client.train)
                #            for client in self.selected_clients]
                # [t.start() for t in threads]
                # [t.join() for t in threads]

                self.receive_models()
                self.aggregate_parameters()

                self.Budget.append(time.time() - s_t)
                # print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            self.calculate_forgetting(task=task, glob_iter=glob_iter)
