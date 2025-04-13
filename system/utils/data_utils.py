import os
import pickle
import time
from typing import Any
import numpy as np
import torch
import torch.utils.data as data
import gc
import torchvision.transforms as transforms

img_size = 32
cifar100_train_transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.RandomCrop((img_size, img_size), padding=4),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.ColorJitter(brightness=0.24705882352941178),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
cifar100_test_transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

# Mean và std của ImageNet1k gốc
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

imagenet_train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((img_size, img_size), padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.24705882352941178),
    # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # Optional
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)])

imagenet_test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_size, img_size)),  
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)])


# client data 1 task
def read_client_data_FCL_imagenet1k(index, task = 0, classes_per_task = 2, count_labels=False):
    
    # datadir = '/root/projects/FCL/dataset/imagenet1k224-classes/'
    datadir = '/root/projects/FCL/dataset/imagenet1k-classes/'
    class_order = np.load('/root/projects/FCL/dataset/class_order/class_order_imagenet1k.npy', allow_pickle=True)

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
        data = np.load(data_file)
        
        x_train.append(data[:train_images_per_class])
        x_test.append(data[train_images_per_class:])
        y_train.append(np.array([_class] * train_images_per_class))
        y_test.append(np.array([_class] * test_images_per_class))
    x_train = torch.tensor(np.concatenate(x_train))
    x_test = torch.tensor(np.concatenate(x_test))
    y_train = torch.from_numpy(np.concatenate(y_train))
    y_test = torch.from_numpy(np.concatenate(y_test))
    return x_train, y_train, x_test, y_test


def load_test_data(datadir, dataset="IMAGENET1k", train_images_per_class=600, test_images_per_class=50):
    if dataset == "CIFAR100":
        classes = list(range(100))
    elif dataset == "IMAGENET1k":
        classes = list(range(1000))
    else:
        raise NotImplementedError("Not supported dataset")

    x_test = None
    y_test = None

    for _class in classes:
        print(f"Loading data for class {_class}...")
        data_file = datadir + str(_class) + '.npy'
        new_x = np.load(data_file)[train_images_per_class:]  # (50, ...)
        new_y = np.full((test_images_per_class,), _class, dtype=np.int64)

        if x_test is None:
            x_test = new_x
            y_test = new_y
        else:
            x_test = np.concatenate((x_test, new_x), axis=0)
            y_test = np.concatenate((y_test, new_y), axis=0)

        del new_x, new_y
        gc.collect()

    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    del x_test, y_test
    gc.collect()

    print(f"Loaded {len(x_test_tensor)} test images and {len(y_test_tensor)} test labels.")
    return x_test_tensor, y_test_tensor


def load_full_data(datadir, dataset="IMAGENET1k", train_images_per_class=600, test_images_per_class=100, concat_every=200):
    if dataset == "CIFAR100":
        classes = list(range(100))
    elif dataset == "IMAGENET1k":
        classes = list(range(1000))
    else:
        raise NotImplementedError("Not supported dataset")

    x_train_all = None
    y_train_all = None
    x_test_all = None
    y_test_all = None

    x_train_list, y_train_list = [], []
    x_test_list, y_test_list = [], []

    for i, _class in enumerate(classes):
        print(f"Loading data full for class {_class}...")
        data_file = datadir + str(_class) + '.npy'
        data = np.load(data_file)  

        new_x_train = data[:train_images_per_class]
        new_x_test = data[train_images_per_class:]  

        new_y_train = np.full((train_images_per_class,), _class, dtype=np.int64)
        new_y_test = np.full((test_images_per_class,), _class, dtype=np.int64)

        x_train_list.append(new_x_train)
        y_train_list.append(new_y_train)
        x_test_list.append(new_x_test)
        y_test_list.append(new_y_test)

        # Mỗi concat_every class thì concat để giảm bộ nhớ
        if (i + 1) % concat_every == 0 or (i + 1) == len(classes):
            print(f"Concatenating batch of {len(x_train_list)} classes...")

            if x_train_all is None:
                x_train_all = np.concatenate(x_train_list, axis=0)
                y_train_all = np.concatenate(y_train_list, axis=0)
                x_test_all = np.concatenate(x_test_list, axis=0)
                y_test_all = np.concatenate(y_test_list, axis=0)
            else:
                x_train_all = np.concatenate([x_train_all] + x_train_list, axis=0)
                y_train_all = np.concatenate([y_train_all] + y_train_list, axis=0)
                x_test_all = np.concatenate([x_test_all] + x_test_list, axis=0)
                y_test_all = np.concatenate([y_test_all] + y_test_list, axis=0)

            # Giải phóng bộ nhớ
            x_train_list.clear()
            y_train_list.clear()
            x_test_list.clear()
            y_test_list.clear()
            gc.collect()

    # Convert sang tensor
    x_train_tensor = torch.tensor(x_train_all, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_all, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test_all, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_all, dtype=torch.long)

    print(f"Loaded {len(x_train_tensor)} training and {len(x_test_tensor)} test images.")
    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor

def get_unique_tasks(task_list):
    unique_tasks = {tuple(sorted(task)) for task in task_list}
    return [list(task) for task in unique_tasks]

# datadir = '/root/projects/FCL/dataset/imagenet1k-classes/'
# load_test_data(datadir, dataset='IMAGENET1k', train_images_per_class=600, test_images_per_class=50)

# test_data_all_task = []
# for task in range(500):    
#     _, test_data, _ = read_client_data_FCL_imagenet1k(1, task=task, classes_per_task=2, count_labels=True)
    
#     test_data_all_task.append(test_data)
#     print(task)