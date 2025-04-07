import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
import numpy as np
import time
import math
from tqdm import tqdm

class network(nn.Module):
    def __init__(self, feature_extractor, num_classes):
        super(network, self).__init__()
        self.feature = feature_extractor
        self.fc = nn.Linear(feature_extractor.fc.in_features, num_classes, bias=True)
        self.feature.fc = nn.Identity()

    def forward(self, input):
        x = self.feature(input)
        x = self.fc(x)
        return x

    # def forward(self, input):
    #     print(f"Input shape: {input.shape}")
    #     x = self.feature(input)
    #     print(f"After feature extractor (self.feature): {x.shape}")
    #     # print("check")
    #     x = self.fc(x)
    #     print(f"After fully connected layer (self.fc): {x.shape}")
    #     return x

    def Incremental_learning(self, numclass):
        weight = self.fc.weight.data
        bias = self.fc.bias.data
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features
        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight
        self.fc.bias.data[:out_feature] = bias

    def feature_extractor(self, inputs):
        return self.feature(inputs)

    def predict(self, fea_input):
        return self.fc(fea_input)
    
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=1, bias=False, dilation=1)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=64):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super().__init__()
        self.inplanes = 64
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.base_width))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=self.base_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def resnet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    

class Teacher(nn.Module):
    def __init__(self, solver, generator, gen_opt, img_shape, iters, class_idx, deep_inv_params, train, args):
        super().__init__()
        self.solver = solver
        self.generator = generator
        self.gen_opt = gen_opt
        self.solver.eval()
        self.generator.eval()
        self.img_shape = img_shape
        self.iters = iters
        self.bn_loss = args.bn_loss
        self.noise = args.noise
        self.ie_loss = args.ie_loss
        self.act_loss = args.act_loss
        self.w_ie = args.w_ie
        self.w_act = args.w_act
        self.di_lr = deep_inv_params[0]
        self.r_feature_weight = deep_inv_params[1]
        self.di_var_scale = deep_inv_params[2]
        self.content_temp = deep_inv_params[3]
        self.content_weight = deep_inv_params[4]
        self.class_idx = list(class_idx)
        self.num_k = len(self.class_idx)
        self.first_time = train
        self.criterion = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss(reduction="none").to('cuda')
        self.smoothing = Gaussiansmoothing(3, 5, 1)
        if self.bn_loss:
            loss_r_feature_layers = []
            for module in self.solver.modules():
                if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.GroupNorm):
                    loss_r_feature_layers.append(DeepInversionFeatureHook(module, 0, self.r_feature_weight))
            self.loss_r_feature_layers = loss_r_feature_layers

    def sample(self, size, return_scores=False):
        self.solver.eval()
        self.generator.train()
        if self.first_time:
            self.first_time = False
            self.get_images(bs=size, epochs=self.iters, idx=-1)
        self.generator.eval()
        with torch.no_grad():
            x_i = self.generator.sample(size, 'cuda')
        with torch.no_grad():
            y_hat = self.solver.forward(x_i)
        y_hat = y_hat[:, self.class_idx]
        _, y = torch.max(y_hat, dim=1)
        return (x_i, y, y_hat) if return_scores else (x_i, y)

    def generate_scores(self, x, allowed_predictions=None, return_label=False):
        self.solver.eval()
        with torch.no_grad():
            y_hat = self.solver.forward(x)
        y_hat = y_hat[:, allowed_predictions]
        _, y = torch.max(y_hat, dim=1)
        return (y, y_hat) if return_label else y_hat

    def generate_scores_pen(self, x):
        self.solver.eval()
        with torch.no_grad():
            y_hat = self.solver.feature_extractor(x)
        return y_hat

    def get_images(self, bs=256, epochs=1000, idx=-1):
        torch.cuda.empty_cache()
        self.generator.train()
        self.generator.to('cuda')
        for epoch in tqdm(range(epochs)):
            inputs = self.generator.sample(bs, 'cuda')
            self.gen_opt.zero_grad()
            self.solver.zero_grad()
            bn_loss = 0
            # content
            if self.act_loss:
                features_t = self.solver.feature(inputs)
                outputs = self.solver.fc(features_t)[:, :self.num_k]
                loss = self.criterion(outputs / self.content_temp, torch.argmax(outputs, dim=1)) * self.content_weight
                loss += - features_t[-1].abs().mean() * self.w_act
            else:
                outputs = self.solver(inputs)[:, :self.num_k]
                ce_loss = self.criterion(outputs / self.content_temp, torch.argmax(outputs, dim=1)) * self.content_weight
                loss = ce_loss

            if self.ie_loss:
                softmax_o_T = F.softmax(outputs, dim=1).mean(dim=0)
                ie_loss = (1.0 + (softmax_o_T * torch.log(softmax_o_T) / math.log(self.num_k)).sum()) * self.w_ie
                loss += ie_loss
            # R_feature loss
            if self.bn_loss:
                for mod in self.loss_r_feature_layers:
                    loss_distr = mod.r_feature * self.r_feature_weight / len(self.loss_r_feature_layers)
                    bn_loss = bn_loss + loss_distr
                loss += bn_loss

            # image prior
            if self.noise:
                inputs_smooth = self.smoothing(F.pad(inputs, (2, 2, 2, 2), mode='reflect'))
                loss_var = self.mse_loss(inputs, inputs_smooth).mean()
                noise_loss = self.di_var_scale * loss_var
                loss += noise_loss
            loss.backward()
            self.gen_opt.step()
        torch.cuda.empty_cache()
        self.generator.eval()

class DeepInversionFeatureHook():
    def __init__(self, module, gram_matrix_weight, layer_weight):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.target = None
        self.gram_matrix_weight = gram_matrix_weight
        self.layer_weight = layer_weight

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False) + 1e-8
        x = module.running_var.data.type(var.type())
        y = module.running_mean.data.type(var.type())
        m1 = torch.log(var**(0.5) / (x + 1e-8)**(0.5)).mean()
        r_feature = m1 - 0.5 * (1.0 - (x + 1e-8 + (y - mean)**2) / var).mean()
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


class Gaussiansmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(Gaussiansmoothing, self).__init__()
        kernel_size = [kernel_size] * dim
        sigma = [sigma] * dim
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1)).to('cuda')
        self.register_buffer('weight', kernel)
        self.groups = channels
        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

    def forward(self, input):
        return self.conv(input, weight=self.weight, groups=self.groups)
    
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