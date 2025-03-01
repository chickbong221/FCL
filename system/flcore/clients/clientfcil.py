import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
from flcore.utils.fcil_utils import entropy, get_one_hot
import torch.optim as optim
from torch.nn import functional as F


class clientFCIL(Client):
    def __init__(self, args, id, train_data, test_data, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_data, test_data, train_samples, test_samples, **kwargs)
        self.start = True
        self.signal = False

        self.memory_size = args.memory_size
        self.task_size = 2      # Check it out later

        self.train_loader = None
        self.current_class = None
        self.last_class = None
        self.task_id_old = -1
        self.last_entropy = 0

    def train(self, ep_g, model_old):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.00001)

        if model_old[1] != None:
            if self.signal:
                self.old_model = model_old[1]
            else:
                self.old_model = model_old[0]
        else:
            if self.signal:
                self.old_model = model_old[0]

        if self.old_model != None:
            print('load old model')
            self.old_model.eval()

        for epoch in range(max_local_epochs):
            loss_cur_sum, loss_mmd_sum = [], []
            if (epoch + ep_g * 20) % 200 == 100:
                if self.num_classes == self.task_size:
                    opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 25, weight_decay=0.00001)
                else:
                    for p in opt.param_groups:
                        p['lr'] = self.learning_rate / 5
            elif (epoch + ep_g * 20) % 200 == 150:
                if self.num_classes > self.task_size:
                    for p in opt.param_groups:
                        p['lr'] = self.learning_rate / 25
                else:
                    opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 25, weight_decay=0.00001)
            elif (epoch + ep_g * 20) % 200 == 180:
                if self.num_classes == self.task_size:
                    opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 125, weight_decay=0.00001)
                else:
                    for p in opt.param_groups:
                        p['lr'] = self.learning_rate / 125
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.cuda(self.device), target.cuda(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                loss_value = self._compute_loss(indexs, images, target)
                opt.zero_grad()
                loss_value.backward()
                opt.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    """
        Compute loss function
    """
    def _compute_loss(self, indexs, imgs, label):
        output = self.model(imgs)

        target = get_one_hot(label, self.num_classes, self.device)
        output, target = output.cuda(self.device), target.cuda(self.device)
        if self.old_model == None:
            w = self.efficient_old_class_weight(output, label)
            loss_cur = torch.mean(w * F.binary_cross_entropy_with_logits(output, target, reduction='none'))

            return loss_cur
        else:
            w = self.efficient_old_class_weight(output, label)
            loss_cur = torch.mean(w * F.binary_cross_entropy_with_logits(output, target, reduction='none'))

            distill_target = target.clone()
            old_target = torch.sigmoid(self.old_model(imgs))
            old_task_size = old_target.shape[1]
            distill_target[..., :old_task_size] = old_target
            loss_old = F.binary_cross_entropy_with_logits(output, distill_target)

            return 0.5 * loss_cur + 0.5 * loss_old

    def beforeTrain(self, task_id_new, group):
        if task_id_new != self.task_id_old:
            self.task_id_old = task_id_new
            self.num_classes = self.task_size * (task_id_new + 1)
            if group != 0:
                if self.current_class != None:
                    self.last_class = self.current_class
                self.current_class = random.sample([x for x in range(self.num_classes - self.task_size, self.num_classes)], 6)
                # print(self.current_class)
            else:
                self.last_class = None

        self.train_loader = self._get_train_and_test_dataloader(self.current_class, False)

    def update_new_set(self):
        self.model = model_to_device(self.model, False, self.device)
        self.model.eval()
        self.signal = False
        self.signal = self.entropy_signal(self.train_loader)

        if self.signal and (self.last_class != None):
            self.learned_numclass += len(self.last_class)
            self.learned_classes += self.last_class

            m = int(self.memory_size / self.learned_numclass)
            self._reduce_exemplar_sets(m)
            for i in self.last_class:
                images = self.train_dataset.get_image_class(i)
                self._construct_exemplar_set(images, m)

        self.model.train()

        self.train_loader = self._get_train_and_test_dataloader(self.current_class, True)