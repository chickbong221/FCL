import os
import pdb
import json
import random
import threading
import numpy as np
import torch

from datetime import datetime

def combine_data(data):
    x, y = [], []
    for i in range(len(data)):
        x.append(data[i][0])
        y.append(data[i][1])
    x, y = torch.cat(x), torch.cat(y)
    return x, y