import torch
import numpy as np
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k
import math, os
from utils_core.LANDER_utils import DataIter, UnlabeledImageDataset


class clientLANDER(Client):
    def __init__(self, args, id, train_data, test_data, **kwargs):
        super().__init__(args, id, train_data, test_data, **kwargs)
        self.syn_data_loader = self.get_syn_data_loader()

    """
    self.total_classes = self.available_labels
    self
    """
    def get_syn_data_loader(self):
        if self.args["dataset"] == "CIFAR100":
            dataset_size = 50000
            num_tasks = 500
        elif self.args["dataset"] == "IMAGENET1k":
            dataset_size = 100000
            num_tasks = 50

        iters = math.ceil(dataset_size / (self.args["num_clients"] * num_tasks * self.args["batch_size"]))
        # syn_bs = int(self.nums / iters)
        syn_bs = self.args["syn_bs"]*self.args["batch_size"]
        data_dir = os.path.join(self.save_dir, "task_{}".format(self._cur_task - 1))
        print("iters{}, syn_bs:{}, data_dir: {}".format(iters, syn_bs, data_dir))
        # print(syn_bs)
        syn_dataset = UnlabeledImageDataset(data_dir, transform=self.transform, nums=self.nums)
        syn_data_loader = torch.utils.data.DataLoader(
            syn_dataset, batch_size=syn_bs, shuffle=True, persistent_workers=True,
            num_workers=self.args["num_worker"], multiprocessing_context=self.args["mulc"])
        return syn_data_loader
    
 
    def _local_finetune(self, teacher, model, task_id, lr):
        alpha = np.log2(self.available_labels / 2 + 1)
        beta = np.sqrt(self.available_labels_current / self.available_labels)
        cur = self.args["cur"] * (1 + 1 / alpha) / beta
        pre = self.args["pre"] * alpha * beta

        model.train()
        teacher.train()
        total_loss = 0 
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=self.args['weight_decay'])
        syn_data_iter = DataIter()
        pass