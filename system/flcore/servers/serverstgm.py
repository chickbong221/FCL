import time
import torch
from flcore.clients.clientstgm import clientSTGM
from flcore.servers.serverbase import Server
from threading import Thread
from utils.model_utils import read_client_data_FCL, read_client_data_FCL_imagenet1k
from torch.optim.lr_scheduler import StepLR
import numpy as np


class FedSTGM(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientSTGM)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.update_grads = None
        self.grad_stgm_c = args.c_parameter
        self.grad_stgm_rounds = args.grad_stgm_rounds
        self.grad_stgm_learning_rate = args.grad_stgm_learning_rate
        self.momentum = args.stgm_momentum
        self.step_size = args.step_size
        self.gamma = args.gamma
        self.device = args.device
        # model_origin = copy.deepcopy(args.model)

    def train(self):

        if self.args.dataset == 'IMAGENET1k':
            N_TASKS = 500
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
                        id, train_data, test_data, label_info = read_client_data_FCL_imagenet1k(i, task=task,
                                                                                                classes_per_task=2,
                                                                                                count_labels=True)
                    else:
                        id, train_data, test_data, label_info = read_client_data_FCL(i, self.data,
                                                                                     dataset=self.args.dataset,
                                                                                     count_labels=True, task=task)

                    # update dataset
                    # assert (self.users[i].id == id)
                    self.clients[i].next_task(train_data, test_data, label_info)  # assign dataloader for new data

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

            for i in range(self.global_rounds):

                glob_iter = i + self.global_rounds * task
                s_t = time.time()
                self.selected_clients = self.select_clients()
                self.send_models()

                if i % self.eval_gap == 0:
                    print(f"\n-------------Round number: {i}-------------")
                    print("\nEvaluate global model")
                    self.evaluate(glob_iter=glob_iter)

                for client in self.selected_clients:
                    client.train()

                # threads = [Thread(target=client.train)
                #            for client in self.selected_clients]
                # [t.start() for t in threads]
                # [t.join() for t in threads]

                self.receive_models()
                self.receive_grads()

                """
                Add aggregate STGM
                """
                grad_ez = sum(p.numel() for p in self.global_model.parameters())
                grads = torch.Tensor(grad_ez, self.num_clients)

                for index, model in enumerate(self.grads):
                    grad2vec2(model, grads, index)

                g = self.aggregate_stgm(grads, self.num_clients)

                # model_origin = copy.deepcopy(self.global_model)
                self.overwrite_grad2(self.global_model, g)
                for param in self.global_model.parameters():
                    param.data += param.grad

                # angle = [self.cos_sim(model_origin, self.global_model, models) for models in self.grads]
                # self.angle_value = statistics.mean(angle)
                #
                # angle_value = []
                # for i in self.grads:
                #     for j in self.grads:
                #         angle_value = [self.cosine_similarity(i, j)]
                #
                # self.grads_angle_value = statistics.mean(angle_value)

                self.Budget.append(time.time() - s_t)
                print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

                if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                    break

            print("\nBest accuracy.")
            # self.print_(max(self.rs_test_acc), max(
            #     self.rs_train_acc), min(self.rs_train_loss))
            print(max(self.rs_test_acc))
            print("\nAverage time cost per round.")
            print(sum(self.Budget[1:]) / len(self.Budget[1:]))

            # self.save_results()
            # self.save_global_model()

            if self.num_new_clients > 0:
                self.eval_new_clients = True
                self.set_new_clients(clientSTGM)
                print(f"\n-------------Fine tuning round-------------")
                print("\nEvaluate new clients")
                self.evaluate(glob_iter=glob_iter)

    def aggregate_stgm(self, grad_vec, num_tasks):

        grads = grad_vec.to(self.device)

        GG = grads.t().mm(grads)
        # to(device)
        scale = (torch.diag(GG) + 1e-4).sqrt().mean()
        GG = GG / scale.pow(2)
        Gg = GG.mean(1, keepdims=True)
        gg = Gg.mean(0, keepdims=True)

        w = torch.zeros(num_tasks, 1, requires_grad=True, device=self.device)
        #         w = torch.zeros(num_tasks, 1, requires_grad=True).to(self.device)

        if num_tasks == 50:
            w_opt = torch.optim.SGD([w], lr=self.grad_stgm_learning_rate * 2, momentum=self.momentum)
        else:
            w_opt = torch.optim.SGD([w], lr=self.grad_stgm_learning_rate, momentum=self.momentum)

        scheduler = StepLR(w_opt, step_size=self.step_size, gamma=self.gamma)

        c = (gg + 1e-4).sqrt() * self.grad_stgm_c

        w_best = None
        obj_best = np.inf
        for i in range(self.grad_stgm_rounds + 1):
            w_opt.zero_grad()
            ww = torch.softmax(w, dim=0)
            obj = ww.t().mm(Gg) + c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
            if obj.item() < obj_best:
                obj_best = obj.item()
                w_best = w.clone()
            if i < self.grad_stgm_rounds:
                obj.backward()
                w_opt.step()
                scheduler.step()

                # Check this scheduler. step()

        ww = torch.softmax(w_best, dim=0)
        gw_norm = (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()

        lmbda = c.view(-1) / (gw_norm + 1e-4)
        g = ((1 / num_tasks + ww * lmbda).view(
            -1, 1).to(grads.device) * grads.t()).sum(0) / (1 + self.grad_stgm_c ** 2)
        return g

    def overwrite_grad2(self, m, newgrad):
        newgrad = newgrad * self.num_clients
        for param in m.parameters():
            # Get the number of elements in the current parameter
            num_elements = param.numel()

            # Extract a slice of new_params with the same number of elements
            param_slice = newgrad[:num_elements]

            # Reshape the slice to match the shape of the current parameter
            param.grad = param_slice.view(param.data.size())

            # Move to the next slice in new_params
            newgrad = newgrad[num_elements:]

def grad2vec2(m, grads, task):
    grads[:, task].fill_(0.0)
    all_params = torch.cat([param.detach().view(-1) for param in m.parameters()])
    # print(all_params.size())
    grads[:, task].copy_(all_params)