import copy
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clientbase import Client
from copy import deepcopy
from flcore.utils_core.FedMFCL_utils import combine_data


class clientMFCL(Client):
    def __init__(self, args, id, train_data, test_data, kd_weight, ft_weight, syn_size, **kwargs):
        super().__init__(args, id, train_data, test_data, **kwargs)
        self.args = args
        self.kd_criterion = nn.MSELoss(reduction='none')
        self.last_valid_dim = 0
        self.valid_dim = 0
        self.mu = kd_weight
        self.ft_weight = ft_weight
        self.syn_size = syn_size

    def train(self, model, lr, teacher, generator_server):
        self.trainloader = self.load_train_data()
        model.to('cuda')
        model.train()
        opt = optim.SGD(model.parameters(), lr=lr, weight_decay=0.00001)
        if teacher is None:
            for epoch in range(self.args.local_epochs):
                for i, (x, y) in enumerate(self.trainloader):
                    x, y = x.to('cuda'), y.to('cuda')
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        else:
            self.train_cl(model, teacher, opt)

    def train_cl(self, model, teacher, opt):
        self.dw_k = torch.ones((self.valid_dim + 1), dtype=torch.float32)
        previous_teacher, previous_linear = deepcopy(teacher[0]), deepcopy(teacher[1])
        for epoch in range(self.args.local_epochs):
            for i, (x, y) in enumerate(self.trainloader):
                x, y = x.to('cuda'), y.to('cuda')
                idx1 = torch.where(y >= self.last_valid_dim)[0]
                x_replay, y_replay, y_replay_hat = self.sample(previous_teacher, self.args.syn_size)
                y_hat = previous_teacher.generate_scores(x, allowed_predictions=np.arange(self.last_valid_dim))
                _, y_hat_com = combine_data(((x, y_hat), (x_replay, y_replay_hat)))
                x_com, y_com = combine_data(((x, y), (x_replay, y_replay)))
                logits_pen = model.feature(x_com)
                logits = model.fc(logits_pen)
                mappings = torch.ones(self.valid_dim, dtype=torch.float32, device='cuda') 
                dw_cls = mappings[y_com.long()]
                loss_class = self.criterion(logits[idx1, self.last_valid_dim:self.valid_dim], (y_com[idx1] - self.last_valid_dim), dw_cls[idx1])
                with torch.no_grad():
                    feat_class = model.feature(x_com).detach()
                loss_class += self.criterion(model.fc(feat_class), y_com, dw_cls) * self.args.w_ft
                loss_kd = self.kd(x_com, previous_linear, logits_pen, previous_teacher)
                total_loss = loss_class + loss_kd
                opt.zero_grad()
                total_loss.backward()
                opt.step()

    def kd(self, x_com, previous_linear, logits_pen, previous_teacher):

        # try:
        #     has_nan = torch.isnan(logits_pen).any()
        #     has_inf = torch.isinf(logits_pen).any()
        #     print(f"kd: logits_pen has NaN: {has_nan}, has Inf: {has_inf}")
        # except RuntimeError as e:
        #     print(f"kd: Failed to check logits_pen: {e}")
            
        kd_index = np.arange(x_com.size(0))
        dw_KD = self.dw_k[-1 * torch.ones(len(kd_index),).long()].to('cuda')
        logits_KD = previous_linear(logits_pen[kd_index])[:, :self.last_valid_dim]
        logits_KD_past = previous_linear(previous_teacher.generate_scores_pen(x_com[kd_index]))[:, :self.last_valid_dim]
        loss_kd = self.args.w_kd * (self.kd_criterion(logits_KD, logits_KD_past).sum(dim=1) * dw_KD).mean() / (logits_KD.size(1))
        return loss_kd

    def sample(self, teacher, dim, return_scores=True):
        return teacher.sample(dim, return_scores=return_scores)

    def criterion(self, logits, targets, data_weights):
        return (F.cross_entropy(logits, targets) * data_weights).mean()
    
    # def criterion(self, logits, targets):
    #     return (F.cross_entropy(logits, targets)).mean()