import os
import pdb
import json
import random
import threading
import numpy as np
import torch
from copy import deepcopy
from flcore.trainmodel.fedMFCL_models import *
from datetime import datetime

def combine_data(data):
    x, y = [], []
    for i in range(len(data)):
        x.append(data[i][0])
        y.append(data[i][1])
    x, y = torch.cat(x), torch.cat(y)
    return x, y

def fedavg_aggregation(weights):
    w_avg = deepcopy(weights[0])
    for k in w_avg.keys():
        for i in range(1, len(weights)):
            w_avg[k] += weights[i][k]
        w_avg[k] = torch.div(w_avg[k], len(weights))
    return w_avg

def train_gen(model, valid_out_dim, generator, args):
    dataset_size = (-1, 3, args.img_size, args.img_size)
    model.to('cuda')
    generator_optimizer = torch.optim.Adam(params=generator.parameters(), lr=args.generator_lr)
    teacher = Teacher(solver=model, generator=generator, gen_opt=generator_optimizer,
                      img_shape=dataset_size, iters=args.pi, deep_inv_params=[1e-3, args.w_bn, args.w_noise, 1e3, 1],
                      class_idx=np.arange(valid_out_dim), train=True, args=args)
    teacher.sample(args.server_ss, return_scores=False)
    return teacher, deepcopy(model.fc)