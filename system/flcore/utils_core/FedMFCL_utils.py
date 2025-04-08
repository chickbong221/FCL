import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flcore.trainmodel.FedMFCL_model import *
from copy import deepcopy

def combine_data(data):
    x, y = [], []
    for i in range(len(data)):
        x.append(data[i][0])
        y.append(data[i][1])
    x, y = torch.cat(x), torch.cat(y)
    return x, y

def train_gen(model, valid_out_dim, generator, args):
    dataset_size = (-1, 3, args.img_size, args.img_size)
    model.to(args.device)
    generator_optimizer = torch.optim.Adam(params=generator.parameters(), lr=args.generator_lr)
    teacher = Teacher(solver=model, generator=generator, gen_opt=generator_optimizer,
                      img_shape=dataset_size, iters=args.pi, deep_inv_params=[1e-3, args.w_bn, args.w_noise, 1e3, 1],
                      class_idx=np.arange(valid_out_dim), train=True, args=args)
    teacher.sample(args.server_ss, return_scores=False)
    return teacher, deepcopy(model.fc)


class Generator(nn.Module):
    def __init__(self, zdim, in_channel, img_sz, out_channel):
        super(Generator, self).__init__()
        self.z_dim = zdim
        self.out_channel = out_channel

        self.init_size = img_sz // 4
        self.l1 = nn.Sequential(nn.Linear(zdim, self.out_channel * self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(self.out_channel),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(self.out_channel, self.out_channel, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channel, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(self.out_channel, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, in_channel, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(in_channel, affine=False)
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], self.out_channel, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img

    def sample(self, size, device):
        # sample z
        z = torch.randn(size, self.z_dim).to(device)
        z = z.to(device)
        X = self.forward(z)
        return X
    

class GeneratorBig(nn.Module):
    def __init__(self, zdim, in_channel, img_sz, convdim):
        super(GeneratorBig, self).__init__()
        self.z_dim = zdim
        self.init_size = img_sz // (2**5)
        self.dim = convdim
        self.l1 = nn.Sequential(nn.Linear(zdim, self.dim * self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(self.dim),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.dim, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.dim, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks3 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.dim, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks4 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.dim, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks5 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.dim, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks6 = nn.Sequential(
            nn.Conv2d(self.dim, in_channel, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(in_channel, affine=False)
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], self.dim, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks3(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks4(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks5(img)
        img = self.conv_blocks6(img)
        return img

    def sample(self, size, device):
        # sample z
        z = torch.randn(size, self.z_dim).to(device)
        z = z.to(device)
        X = self.forward(z)
        return X