import numpy as np
import time
import copy
import torch
import copy
from utils.model_utils import ParamDict
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from flcore.clients.clientbase import Client


class clientSTGM(Client):
    def __init__(self, args, id, train_data, **kwargs):
        super().__init__(args, id, train_data, **kwargs)

    def train(self, task=None):
        trainloader = self.load_train_data(task=task)
        self.model.train()
        old_model = copy.deepcopy(self.model)

        start_time = time.time()

        max_local_epochs = self.local_epochs
        inner_models = []
        """ ============ Current Task ============  """
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if not self.args.tgm:
            """ ======== Approximate Last Task ========  """
            for task in self.task_dict:
                trainloader = self.load_train_data(task=task)
                for epoch in range(max_local_epochs):
                    for i, (x, y) in enumerate(trainloader):
                        pass

            """ ===== Temporal Gradient Matching ======  """
            meta_weights = self.tgm_high(
                meta_weights=self.model,
                inner_weights=inner_models,
                lr_meta=self.stgm_meta_lr
            )
            self.model.load_state_dict(copy.deepcopy(meta_weights))
        else:
            pass

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def tgm_high(self, meta_weights, inner_weights, lr_meta):
        """
        Input:
        - meta_weights: class X(nn.Module)
        - inner_weights: list[X(nn.Module), X(nn.Module), ..., X(nn.Module)]
        - lr_meta: scalar value

        Output:
        - meta_weights: class X(nn.Module)

        """
        all_domain_grads = []
        flatten_meta_weights = torch.cat([param.view(-1) for param in meta_weights.parameters()])
        for i_domain in range(self.num_clients):
            domain_grad_diffs = [torch.flatten(inner_param - meta_param) for inner_param, meta_param in
                                 zip(inner_weights[i_domain].parameters(), meta_weights.parameters())]
            domain_grad_vector = torch.cat(domain_grad_diffs)
            all_domain_grads.append(domain_grad_vector)

        """
        - Grads normalization.
        """
        if self.grad_balance:
            # Apply balancing
            # Step 1: Compute norms for each gradient vector
            domain_grad_norms = [torch.norm(grad) for grad in all_domain_grads]

            # Step 2: Determine scaling factors to balance the norms
            # Example: Scale all norms to a target value (e.g., the average norm)
            target_norm = torch.mean(torch.tensor(domain_grad_norms))
            scaling_factors = [target_norm / norm if norm > 0 else 1.0 for norm in domain_grad_norms]

            # Step 3: Scale gradient vectors
            balanced_retain_grads = [grad * scale for grad, scale in zip(domain_grad_norms, scaling_factors)]

            # Step 4: Stack the balanced gradients into a tensor
            all_domains_grad_tensor = torch.stack(balanced_retain_grads).t()
        else:
            all_domains_grad_tensor = torch.stack(all_domain_grads).t()

        all_domains_grad_tensor = torch.stack(all_domain_grads).t()

        # print(all_domains_grad_tensor)
        g = self.tgm_low(all_domains_grad_tensor, self.num_clients)

        flatten_meta_weights += g * lr_meta

        vector_to_parameters(flatten_meta_weights, meta_weights.parameters())
        meta_weights = ParamDict(meta_weights.state_dict())

        return meta_weights

    def tgm_low(self, grad_vec, num_tasks):

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
            w_opt = torch.optim.SGD([w], lr=self.stgm_learning_rate * 2, momentum=self.stgm_momentum)
        else:
            w_opt = torch.optim.SGD([w], lr=self.stgm_learning_rate, momentum=self.stgm_momentum)

        scheduler = StepLR(w_opt, step_size=self.stgm_step_size, gamma=self.stgm_gamma)

        c = (gg + 1e-4).sqrt() * self.stgm_c

        w_best = None
        obj_best = np.inf
        for i in range(self.stgm_rounds + 1):
            w_opt.zero_grad()
            ww = torch.softmax(w, dim=0)
            obj = ww.t().mm(Gg) + c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
            if obj.item() < obj_best:
                obj_best = obj.item()
                w_best = w.clone()
            if i < self.stgm_rounds:
                obj.backward(retain_graph=True)
                w_opt.step()
                scheduler.step()

                # Check this scheduler. step()

        ww = torch.softmax(w_best, dim=0)
        gw_norm = (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()

        lmbda = c.view(-1) / (gw_norm + 1e-4)
        g = ((1 / num_tasks + ww * lmbda).view(
            -1, 1).to(grads.device) * grads.t()).sum(0) / (1 + self.stgm_c ** 2)
        return g