import os
import pickle
from typing import Any
import numpy as np
import torch
import torch.utils.data as data


# client data 1 task
def read_client_data_FCL_imagenet1k(index, task = 0, classes_per_task = 2, count_labels=False):
    
    datadir = 'dataset/imagenet1k-classes/'
    class_order = np.load('dataset/class_order/class_order_imagenet1k.npy', allow_pickle=True)
    class_order = class_order[index]

    x_train, y_train, x_test, y_test = load_data(datadir, class_order[task*classes_per_task:(task+1)*classes_per_task])
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


def read_client_data_FCL_cifar100(index, task = 0, classes_per_task = 2, count_labels=False):
    
    datadir = 'dataset/cifar100-classes/'
    class_order = np.load('dataset/class_order/class_order_cifar100.npy', allow_pickle=True)
    class_order = class_order[index]

    x_train, y_train, x_test, y_test = load_data(datadir, class_order[task*classes_per_task:(task+1)*classes_per_task], train_images_per_class=500, test_images_per_class=100)
    x_train, x_test = x_train.type(torch.FloatTensor), x_test.type(torch.FloatTensor)
    y_train, y_test = torch.Tensor(y_train.type(torch.long)), torch.Tensor(y_test.type(torch.long))
    
    # img_size = 32
    # train_transform = transforms.Compose([transforms.RandomCrop((img_size, img_size), padding=4),
    #                             transforms.RandomHorizontalFlip(p=0.5),
    #                             transforms.ColorJitter(brightness=0.24705882352941178),
    #                             # transforms.ToTensor(),
    #                             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    # test_transform = transforms.Compose([transforms.Resize(img_size), 
    #                                         # transforms.ToTensor(), 
    #                                         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    # train_data = Transform_dataset(x_train, y_train, train_transform)
    # test_data = Transform_dataset(x_test, y_test, test_transform)

    train_data = Transform_dataset(x_train, y_train)
    test_data = Transform_dataset(x_test, y_test)

    if count_labels:
        label_info = {}
        unique_y, counts=torch.unique(y_train, return_counts=True)
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
    
def load_data(datadir, classes=[], train_images_per_class = 600, test_images_per_class = 50):
    x_train, y_train, x_test, y_test = [], [], [], []
    
    for _class in classes:
        data_file = datadir + str(_class) + '.npy'
        # print(f"Loading data from {data_file}")
        # if os.path.getsize(data_file) == 0:
        #     raise ValueError(f"File {data_file} is empty.")
        # else:
        #     print(os.path.getsize(data_file))
        new_x = np.load(data_file)
        # print(new_x[0].shape)
        x_train.append(new_x[:train_images_per_class])
        x_test.append(new_x[train_images_per_class:])
        y_train.append(np.array([_class] * train_images_per_class))
        y_test.append(np.array([_class] * test_images_per_class))
    x_train = torch.tensor(np.concatenate(x_train))
    x_test = torch.tensor(np.concatenate(x_test))
    y_train = torch.from_numpy(np.concatenate(y_train))
    y_test = torch.from_numpy(np.concatenate(y_test))
    return x_train, y_train, x_test, y_test

def load_test_data(datadir, dataset="IMAGENET1k", train_images_per_class = 600, test_images_per_class = 50):
    x_test, y_test = [], []
    
    if dataset == "CIFAR100":
        classes = list(range(100))
    elif dataset == "IMAGENET1k":
        classes = list(range(1000))
    else:
        raise NotImplementedError("Not supported dataset")
    
    for _class in classes:
        print(f"Loading data from {datadir}{_class}.npy")
        data_file = datadir + str(_class) + '.npy'
        new_x = np.load(data_file)
        x_test.append(new_x[train_images_per_class:])
        y_test.append(np.array([_class] * test_images_per_class))

    return x_test, y_test

def get_unique_tasks(task_list):
    unique_tasks = {tuple(sorted(task)) for task in task_list}
    return [list(task) for task in unique_tasks]


