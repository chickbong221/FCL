import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import random


def get_one_hot(target, num_class, device):
    one_hot=torch.zeros(target.shape[0],num_class).cuda(device)
    one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot

def entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy