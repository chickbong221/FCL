import torch
from flcore.clients.clientLANDER import clientLANDER
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k
from utils_core.LANDER_utils import NLGenerator, NLGenerator_IN
import time
import copy
import numpy as np
import torch.nn.init as init
from torch import nn

class LANDER(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientLANDER)
        self.old_network = self.global_model.copy().freeze()

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def data_generation(self):
        if self.args.dataset == 'CIFAR100':
            img_size = 32
            img_shape = (3, 32, 32)
            generator = NLGenerator(ngf=64, img_size=img_size, nc=3, nl=10,
                                    label_emb=self.label_emb, le_emb_size=self.args['nz'],
                                    sbz=self.args['synthesis_batch_size'])
        elif self.args.dataset == 'IMAGENET1k':
            img_size = 224
            img_shape = (3, 224, 224)
            generator = NLGenerator_IN(ngf=64, img_size=img_size, nc=3, nl=10,
                                      label_emb=self.label_emb, le_emb_size=self.args['nz'],
                                      sbz=self.args['synthesis_batch_size'])
            
        student = copy.deepcopy(self.global_model)
        student.apply(weight_init)
            
            
        pass

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

                for i, u in enumerate(self.clients):
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)
                    # print("task 0")
                    # print("client", i)
                    # print("available_label", u.available_labels)
                    # print("available_label_current", u.available_labels_current)
                    # print("available_label_past", u.available_labels_past)

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
            self.global_model.update_fc(self.clients[0].available_labels)
            self.best_model = None
            self.lowest_loss = np.inf
            optimizer = torch.optim.SGD(self.global_model.parameters(), lr=self.args['local_lr'], momentum=0.9, weight_decay=self.args['weight_decay'])
            if self.args['dataset'] == "IMANGENET1k":
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args["global_rounds"], eta_min=1e-3)

            if self.current_task + 1 != self.N_TASKS:
                self.data_generation()

            for i in range(self.global_rounds):
                local_weights = []
                loss_weight = []
                glob_iter = i + self.global_rounds * task
                s_t = time.time()

                self.selected_clients = self.select_clients()
                self.send_models()

                if i%self.eval_gap == 0:
                    print(f"\n-------------Round number: {i}-------------")
                    self.eval(task=task, glob_iter=glob_iter, flag="global")

                for idx, client in enumerate(self.select_clients):
                    if task == 0 :
                        w, total_loss = client._local_update(copy.deepcopy(self.global_model))
                    else:
                        w, total_syn, total_local, total_loss = client._local_finetune(self.old_network,
                                                                                        copy.deepcopy(self.global_model),
                                                                                        self.current_task, 
                                                                                        scheduler.get_last_lr()[0])
                    
                    local_weights.append(copy.deepcopy(w))
                    loss_weight.append(total_loss)
                    del w
                    torch.cuda.empty_cache()
                scheduler.step()
                sum_loss = sum(loss_weight)
                if sum_loss < self.lowest_loss:
                    self.lowest_loss = sum_loss
                    self.best_model = copy.deepcopy(self.global_model.state_dict())

                self.receive_models()
                self.aggregate_parameters()

                if i%self.eval_gap == 0:
                    self.eval(task=task, glob_iter=glob_iter, flag="local")

                self.Budget.append(time.time() - s_t)

            self.eval_task(task=task, glob_iter=glob_iter, flag="local")
            
            # need eval before data update
            self.send_models()
            self.eval_task(task=task, glob_iter=glob_iter, flag="global")

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)




        