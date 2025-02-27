import time
import torch
from flcore.clients.clientfcil import clientFCIL
from flcore.servers.serverbase import Server
from threading import Thread
from utils.model_utils import read_client_data_FCL, read_client_data_FCL_imagenet1k


class FedFCIL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFCIL)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.pool_grad = None
        self.best_model_1 = None
        self.best_model_2 = None
        self.best_perf = 0
        self.encode_model = encode_model                  # Check this
        self.monitor_dataset = Proxy_Data(test_transform) # Check this

    def train(self):

        if self.args.dataset == 'IMAGENET1k':
            N_TASKS = 500
        else:
            N_TASKS = len(self.data['train_data'][self.data['client_names'][0]]['x'])
        print(str(N_TASKS) + " tasks are available")

        """
            Init for parameters for learning FCIL 
        """
        old_client_0 = []
        old_client_1 = [i for i in range(args.num_clients)]
        new_client = []
        models = []

        classes_learned = args.task_size # Verify later
        old_task_id = -1

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
                    # vassert (self.users[i].id == id)
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

                """
                    L85-L103 FCIL/fl_main.py
                    - model_g -> global_model
                    - proxy_server -> ? 
                    - 
                """
                pool_grad = []
                model_old = proxy_server.model_back()
                task_id = task  # ep_g // args.tasks_global (exchange with this)
                ep_g = (task*self.global_rounds + i)

                if task_id != old_task_id and old_task_id != -1:
                    overall_client = len(old_client_0) + len(old_client_1) + len(new_client)
                    new_client = [i for i in range(overall_client, overall_client + args.task_size)]
                    old_client_1 = random.sample([i for i in range(overall_client)], int(overall_client * 0.9))
                    old_client_0 = [i for i in range(overall_client) if i not in old_client_1]
                    num_clients = len(new_client) + len(old_client_1) + len(old_client_0)
                    print(old_client_0)

                if task_id != old_task_id and old_task_id != -1:
                    classes_learned += args.task_size
                    model_g.Incremental_learning(classes_learned)
                    model_g = model_to_device(model_g, False, args.device)

                print('federated global round: {}, task_id: {}'.format(ep_g, task_id))
                w_local = []

                self.selected_clients = self.select_clients()
                self.send_models()

                if i % self.eval_gap == 0:
                    print(f"\n-------------Round number: {i}-------------")
                    print("\nEvaluate global model")
                    self.evaluate(glob_iter=glob_pool_graditer)

                for client in self.selected_clients:
                    if index in old_client:
                        client.beforeTrain(task_id, 0)
                    else:
                        client.beforeTrain(task_id, 1)

                    client.update_new_set()
                    print(client.signal)
                    client.train(ep_g, model_old)
                    local_model = client.model.state_dict()
                    proto_grad = client.proto_grad_sharing()

                    print('*' * 60)
                    """
                        L106-110 comes here
                        - returns client.model + client.proto_grad
                        - grad_i in proto_grad? what is the shape of proto_grad?
                        - append to pool_grad
                    """
                    # local_model, proto_grad = local_train(models, c, model_g, task_id, model_old, ep_g, old_client_0)
                    local_model, proto_grad = client.train(task_id, model_old, ep_g, old_client_0)
                    w_local.append(local_model)
                    if proto_grad != None:
                        for grad_i in proto_grad:
                            pool_grad.append(grad_i)

                # threads = [Thread(target=client.train)
                #            for client in self.selected_clients]
                # [t.start() for t in threads]
                # [t.join() for t in threads]

                self.receive_models()
                if self.dlg_eval and i % self.dlg_gap == 0:
                    self.call_dlg(i)
                self.aggregate_parameters()

                """
                    L121-L124 comes here
                    - proxy_server.dataloader(pool_grad) do for what?
                """
                w_g_last = copy.deepcopy(model_g.state_dict())

                model_g.load_state_dict(w_g_new)

                proxy_server.model = copy.deepcopy(model_g)
                proxy_server.dataloader(pool_grad)

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
                self.set_new_clients(clientFCIL)
                print(f"\n-------------Fine tuning round-------------")
                print("\nEvaluate new clients")
                self.evaluate(glob_iter=glob_iter)

    def model_back(self):
        return [self.best_model_1, self.best_model_2]

    def dataloader(self, pool_grad):
        self.pool_grad = pool_grad
        if len(pool_grad) != 0:
            self.reconstruction()
            self.monitor_dataset.getTestData(self.new_set, self.new_set_label)
            self.monitor_loader = DataLoader(dataset=self.monitor_dataset, shuffle=True, batch_size=64, drop_last=True)
            self.last_perf = 0
            self.best_model_1 = self.best_model_2

        cur_perf = self.monitor()
        print(cur_perf)
        if cur_perf >= self.best_perf:
            self.best_perf = cur_perf
            self.best_model_2 = copy.deepcopy(self.model)

    """
        Verify later
    """
    def monitor(self):
        self.model.eval()
        correct, total = 0, 0
        for step, (imgs, labels) in enumerate(self.monitor_loader):
            imgs, labels = imgs.cuda(self.device), labels.cuda(self.device)
            with torch.no_grad():
                outputs = self.model(imgs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = 100 * correct / total

        return accuracy

    def gradient2label(self):
        pool_label = []
        for w_single in self.pool_grad:
            pred = torch.argmin(torch.sum(w_single[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)
            pool_label.append(pred.item())

        return pool_label

    def reconstruction(self):
        self.new_set, self.new_set_label = [], []

        tt = transforms.Compose([transforms.ToTensor()])
        tp = transforms.Compose([transforms.ToPILImage()])
        pool_label = self.gradient2label()
        pool_label = np.array(pool_label)
        # print(pool_label)
        class_ratio = np.zeros((1, 100))

        for i in pool_label:
            class_ratio[0, i] += 1

        for label_i in range(100):
            if class_ratio[0, label_i] > 0:
                num_augmentation = self.num_image
                augmentation = []

                grad_index = np.where(pool_label == label_i)
                for j in range(len(grad_index[0])):
                    # print('reconstruct_{}, {}-th'.format(label_i, j))
                    grad_truth_temp = self.pool_grad[grad_index[0][j]]

                    dummy_data = torch.randn((1, 3, 32, 32)).to(self.device).requires_grad_(True)
                    label_pred = torch.Tensor([label_i]).long().to(self.device).requires_grad_(False)

                    optimizer = torch.optim.LBFGS([dummy_data, ], lr=0.1)
                    criterion = nn.CrossEntropyLoss().to(self.device)

                    recon_model = copy.deepcopy(self.encode_model)
                    recon_model = model_to_device(recon_model, False, self.device)

                    for iters in range(self.Iteration):
                        def closure():
                            optimizer.zero_grad()
                            pred = recon_model(dummy_data)
                            dummy_loss = criterion(pred, label_pred)

                            dummy_dy_dx = torch.autograd.grad(dummy_loss, recon_model.parameters(), create_graph=True)

                            grad_diff = 0
                            for gx, gy in zip(dummy_dy_dx, grad_truth_temp):
                                grad_diff += ((gx - gy) ** 2).sum()
                            grad_diff.backward()
                            return grad_diff

                        optimizer.step(closure)
                        current_loss = closure().item()

                        if iters == self.Iteration - 1:
                            print(current_loss)

                        if iters >= self.Iteration - self.num_image:
                            dummy_data_temp = np.asarray(tp(dummy_data.clone().squeeze(0).cpu()))
                            augmentation.append(dummy_data_temp)

                self.new_set.append(augmentation)
                self.new_set_label.append(label_i)