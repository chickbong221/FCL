import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientFCIL(Client):
    def __init__(self, args, id, train_data, test_data, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_data, test_data, train_samples, test_samples, **kwargs)

    def traintrain(self, ep_g, model_old):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        """
        
        """
        self.model = model_to_device(self.model, False, self.device)
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
            self.old_model = model_to_device(self.old_model, False, self.device)
            self.old_model.eval()

        for epoch in range(max_local_epochs):
            loss_cur_sum, loss_mmd_sum = [], []
            if (epoch + ep_g * 20) % 200 == 100:
                if self.numclass == self.task_size:
                    opt = self.optimizer
                else:
                    for p in opt.param_groups:
                        p['lr'] = self.learning_rate / 5
            elif (epoch + ep_g * 20) % 200 == 150:
                if self.numclass > self.task_size:
                    for p in opt.param_groups:
                        p['lr'] = self.learning_rate / 25
                else:
                    opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 25, weight_decay=0.00001)
            elif (epoch + ep_g * 20) % 200 == 180:
                if self.numclass == self.task_size:
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

        """

        """

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
