import os
import sys
import torch
import wandb
import glog as logger
import numpy as np
import csv
import copy
import time
import random
from utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k
from flcore.metrics.average_forgetting import metric_average_forgetting

class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.global_accuracy_matrix = []
        self.local_accuracy_matrix = []

        if self.args.dataset == 'IMAGENET1k':
            self.N_TASKS = 500
        elif self.args.dataset == 'CIFAR100':
            self.N_TASKS = 50
        else:
            raise NotImplementedError("Not supported dataset")

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
                        train_slow=train_slow,
                        send_slow=send_slow)
            self.clients.append(client)

            # update classes so far & current labels
            client.classes_so_far.extend(label_info['labels'])
            client.current_labels.extend(label_info['labels'])
            client.task_dict[0] = label_info['labels']

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += len(client.train_data)
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(len(client.train_data))
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def test_metrics(self, task, glob_iter, flag):
        
        num_samples = []
        tot_correct = []
        for c in self.clients:
            ct, ns = c.test_metrics(task=task)
            # print(f"task {task}, client {c.id}, correct: {ct}")
            # print(f"num_sample: {ns}")
            tot_correct.append(ct*1.0)
            num_samples.append(ns)

            if flag == "global":
                test_acc = sum(tot_correct)*1.0 / sum(num_samples)

                if self.args.wandb:
                    wandb.log({
                        f"Client_Global/Client_{c.id}/Averaged Test Accurancy": test_acc,
                    }, step=glob_iter)

            if flag == "local":
                test_acc = sum(tot_correct)*1.0 / sum(num_samples)

                if self.args.wandb:
                    wandb.log({
                        f"Client_Local/Client_{c.id}/Averaged Test Accurancy": test_acc,
                    }, step=glob_iter)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct

    def train_metrics(self):

        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def eval(self, task, glob_iter, flag):

        stats = self.test_metrics(task, glob_iter, flag=flag)
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])

        if flag == "global":
            if self.args.wandb:
                wandb.log({
                    "Global/Averaged Train Loss": train_loss,
                    "Global/Averaged Test Accurancy": test_acc,
                }, step=glob_iter)

            # print("Averaged Train Loss: {:.4f}".format(train_loss))
            # print("Averaged Test Accurancy: {:.4f}".format(test_acc))

        if flag == "local":
            if self.args.wandb:
                wandb.log({
                    "Local/Averaged Train Loss": train_loss,
                    "Local/Averaged Test Accurancy": test_acc,
                }, step=glob_iter)

            # print("Averaged Client Train Loss: {:.4f}".format(train_loss))
            # print("Averaged Client Test Accurancy: {:.4f}".format(test_acc))

    # evaluate after end 1 task
    def eval_task(self, task, glob_iter, flag):

        accuracy_on_all_task = []

        for t in range(self.N_TASKS):
            stats = self.test_metrics(task=t, glob_iter=glob_iter, flag="off")
            test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
            accuracy_on_all_task.append(test_acc)

        if flag == "global":
            self.global_accuracy_matrix.append(accuracy_on_all_task)
            forgetting = metric_average_forgetting(task, self.global_accuracy_matrix)

            if self.args.wandb:
                wandb.log({
                    "Global/Averaged Forgetting": forgetting
                }, step=glob_iter)

            print("Global Averaged Forgetting: {:.4f}".format(forgetting))

            csv_filename = f"{self.args.algorithm}_global_accuracy_matrix.csv"
            if os.path.exists(csv_filename):
                os.remove(csv_filename)  # Xóa file cũ

            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(self.global_accuracy_matrix)

        if flag == "local":
            self.local_accuracy_matrix.append(accuracy_on_all_task)
            forgetting = metric_average_forgetting(task, self.local_accuracy_matrix)

            if self.args.wandb:
                wandb.log({
                    "Local/Averaged Forgetting": forgetting
                }, step=glob_iter)

            print("Local Averaged Forgetting: {:.4f}".format(forgetting))

            csv_filename = f"{self.args.algorithm}_local_accuracy_matrix.csv"
            if os.path.exists(csv_filename):
                os.remove(csv_filename)  # Xóa file cũ

            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(self.local_accuracy_matrix)