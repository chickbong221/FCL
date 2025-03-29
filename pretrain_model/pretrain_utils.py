import numpy as np
import torch
import numpy as np
import pickle
import torch.utils.data as data
from typing import Any

def read_all_client_data_FCL_imagenet1k_task_single(num_clients, task = 0, classes_per_task = 2, count_labels=False):
    
    datadir = 'dataset/imagenet1k-classes/'
    class_all_clients = set()
    class_order_ori = np.load('dataset/class_order/class_order_imagenet1k.npy', allow_pickle=True)

    for id in range(num_clients):    
        class_order = class_order_ori[id]
        class_all_clients.update(class_order[task*classes_per_task:(task+1)*classes_per_task])
    class_all_clients = list(class_all_clients)
    class_all_clients.sort()

    x_train, y_train, x_test, y_test = load_data(datadir, class_all_clients)
    x_train, x_test = x_train.type(torch.FloatTensor), x_test.type(torch.FloatTensor)
    y_train, y_test = torch.Tensor(y_train.type(torch.long)), torch.Tensor(y_test.type(torch.long))
    train_data = Transform_dataset(x_train, y_train)
    test_data = Transform_dataset(x_test, y_test)

    if count_labels:
        label_info = {}
        unique_y, counts=torch.unique(y_train, return_counts=True)
        # print("unique_y: " + str(unique_y))
        unique_y=unique_y.detach().numpy()
        counts=counts.detach().numpy()
        label_info['labels']=unique_y
        label_info['counts']=counts

        return train_data, test_data, label_info
    
    return train_data, test_data

def read_all_client_data_FCL_imagenet1k_task_sofar(num_clients, task = 0, classes_per_task = 2, count_labels=False):
    
    datadir = 'dataset/imagenet1k-classes/'
    class_all_clients = set()
    class_order_ori = np.load('dataset/class_order/class_order_imagenet1k.npy', allow_pickle=True)

    for task_in_loop in list(range(task+1)):
        for id in range(num_clients):    
            class_order = class_order_ori[id]
            class_all_clients.update(class_order[task_in_loop*classes_per_task:(task_in_loop+1)*classes_per_task])

    class_all_clients = list(class_all_clients)
    class_all_clients.sort()

    x_train, y_train, x_test, y_test = load_data(datadir, class_all_clients)
    x_train, x_test = x_train.type(torch.FloatTensor), x_test.type(torch.FloatTensor)
    y_train, y_test = torch.Tensor(y_train.type(torch.long)), torch.Tensor(y_test.type(torch.long))
    train_data = Transform_dataset(x_train, y_train)
    test_data = Transform_dataset(x_test, y_test)

    if count_labels:
        label_info = {}
        unique_y, counts=torch.unique(y_train, return_counts=True)
        # print(f"unique_y: {unique_y}")
        # print(f"unique_y_len: {len(unique_y)}")
        unique_y=unique_y.detach().numpy()
        counts=counts.detach().numpy()
        label_info['labels']=unique_y
        label_info['counts']=counts

        return train_data, test_data, label_info
    
    return train_data, test_data


def read_all_client_data_FCL_cifar100_task_single(num_clients, task = 0, classes_per_task = 2, count_labels=False):
    
    datadir = 'dataset/cifar100-classes/'
    class_all_clients = set()
    class_order_ori = np.load('dataset/class_order/class_order_cifar100.npy', allow_pickle=True)

    for id in range(num_clients):    
        class_order = class_order_ori[id]
        class_all_clients.update(class_order[task*classes_per_task:(task+1)*classes_per_task])
    class_all_clients = list(class_all_clients)
    class_all_clients.sort()

    x_train, y_train, x_test, y_test = load_data(datadir, class_all_clients)
    x_train, x_test = x_train.type(torch.FloatTensor), x_test.type(torch.FloatTensor)
    y_train, y_test = torch.Tensor(y_train.type(torch.long)), torch.Tensor(y_test.type(torch.long))
    train_data = Transform_dataset(x_train, y_train)
    test_data = Transform_dataset(x_test, y_test)

    if count_labels:
        label_info = {}
        unique_y, counts=torch.unique(y_train, return_counts=True)
        print("unique_y: " + str(unique_y))
        unique_y=unique_y.detach().numpy()
        counts=counts.detach().numpy()
        label_info['labels']=unique_y
        label_info['counts']=counts

        return train_data, test_data, label_info
    
    return train_data, test_data

def read_all_client_data_FCL_cifar100_task_sofar(num_clients, task = 0, classes_per_task = 2, count_labels=False):
    
    datadir = 'dataset/cifar100-classes/'
    class_all_clients = set()
    class_order_ori = np.load('dataset/class_order/class_order_cifar100.npy', allow_pickle=True)

    for task_in_loop in list(range(task+1)):
        for id in range(num_clients):    
            class_order = class_order_ori[id]
            class_all_clients.update(class_order[task_in_loop*classes_per_task:(task_in_loop+1)*classes_per_task])

    class_all_clients = list(class_all_clients)
    class_all_clients.sort()

    x_train, y_train, x_test, y_test = load_data(datadir, class_all_clients)
    x_train, x_test = x_train.type(torch.FloatTensor), x_test.type(torch.FloatTensor)
    y_train, y_test = torch.Tensor(y_train.type(torch.long)), torch.Tensor(y_test.type(torch.long))
    train_data = Transform_dataset(x_train, y_train)
    test_data = Transform_dataset(x_test, y_test)

    if count_labels:
        label_info = {}
        unique_y, counts=torch.unique(y_train, return_counts=True)
        print(f"unique_y: {unique_y}")
        print(f"unique_y_len: {len(unique_y)}")
        unique_y=unique_y.detach().numpy()
        counts=counts.detach().numpy()
        label_info['labels']=unique_y
        label_info['counts']=counts

        return train_data, test_data, label_info
    
    return train_data, test_data

class Transform_dataset(data.Dataset):
    def __init__(self, X, Y, transform=None) -> None:
        super().__init__()
        self.X = X
        self.Y = Y
        self.transform = transform
    
    def __getitem__(self, index: Any) -> Any:
        x = self.X[index]
        y = self.Y[index]
        if self.transform:
            x = self.transform(x)
        return x,y

    def __len__(self) -> int:
        return len(self.X)
    
def load_data(datadir, classes=[], train_images_per_class = 600, test_images_per_class = 100):
    x_train, y_train, x_test, y_test = [], [], [], []
    for idx, _class in enumerate(classes):
        data_file = datadir + str(_class) + '.npy'
        new_x = np.load(data_file)
        x_train.append(new_x[:train_images_per_class])
        x_test.append(new_x[train_images_per_class:])
        y_train.append(np.array([_class] * train_images_per_class))
        y_test.append(np.array([_class] * test_images_per_class))
    x_train = torch.tensor(np.concatenate(x_train))
    x_test = torch.tensor(np.concatenate(x_test))
    y_train = torch.from_numpy(np.concatenate(y_train))
    y_test = torch.from_numpy(np.concatenate(y_test))
    return x_train, y_train, x_test, y_test

# read_all_client_data_FCL_cifar100_task_sofar(num_clients=10, task=2, count_labels=True)