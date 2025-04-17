import torch
from flcore.clients.clientLANDER import clientLANDER
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k
from utils_core.LANDER_utils import NLGenerator, NLGenerator_IN, UnlabeledImageDataset, save_image_batch, kldiv, custom_cross_entropy, weight_init, get_norm_and_transform
import time, os
import copy
import numpy as np
import torch.nn.init as init
from torch import nn
import math
from PIL import Image
from kornia import augmentation
from torchvision import transforms
from tqdm import tqdm
from torch.nn import functional as F

bn_mmt = 0.9
T = 20.0
student_train_step = 50

class ImagePool(object):
    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._idx = 0

    def add(self, imgs, targets=None):
        save_image_batch(imgs, os.path.join(self.root, "%d.png" % (self._idx)), pack=False)
        self._idx += 1

    def get_dataset(self, nums=None, transform=None, labeled=True):
        return UnlabeledImageDataset(self.root, transform=transform, nums=nums)

class DeepInversionHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module, mmt_rate):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.mmt_rate = mmt_rate
        self.mmt = None
        self.tmp_val = None

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        if self.mmt is None:
            r_feature = torch.norm(module.running_var.data - var, 2) + \
                        torch.norm(module.running_mean.data - mean, 2)
        else:
            mean_mmt, var_mmt = self.mmt
            r_feature = torch.norm(module.running_var.data - (1 - self.mmt_rate) * var - self.mmt_rate * var_mmt, 2) + \
                        torch.norm(module.running_mean.data - (1 - self.mmt_rate) * mean - self.mmt_rate * mean_mmt, 2)

        self.r_feature = r_feature
        self.tmp_val = (mean, var)

    def update_mmt(self):
        mean, var = self.tmp_val
        if self.mmt is None:
            self.mmt = (mean.data, var.data)
        else:
            mean_mmt, var_mmt = self.mmt
            self.mmt = (self.mmt_rate * mean_mmt + (1 - self.mmt_rate) * mean.data,
                        self.mmt_rate * var_mmt + (1 - self.mmt_rate) * var.data)

    def remove(self):
        self.hook.remove()

class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)

class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)

    def next(self):
        try:
            data = next(self._iter)
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next(self._iter)
        return data
    
class NAYER():
    def __init__(self, teacher, student, generator, num_classes, img_size, iterations=100, lr_g=0.1, label_emb=None,
                 synthesis_batch_size=128, adv=0.0, bn=1, oh=1, r=1e-1, ltc=0.2, save_dir='run/fast', transform=None,
                 normalizer=None, device="gpu:0", warmup=10, bn_mmt=0, args=None):
        super(NAYER, self).__init__()
        self.teacher = teacher
        self.student = student
        self.save_dir = save_dir
        self.img_size = img_size
        self.iterations = iterations
        self.lr_g = lr_g
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.args = args
        self.label_emb = label_emb
        self.ltc = ltc
        self.r = r
        self.device = device

        self.num_classes = num_classes
        self.synthesis_batch_size = synthesis_batch_size
        self.normalizer = normalizer

        self.data_pool = ImagePool(root=self.save_dir)
        self.transform = transform
        self.generator = generator.cuda().train()
        self.ep = 0
        self.ep_start = self.args['warmup'] + 1 # need more time to the student train well.
        self.prev_z = None

        if self.args["dataset"] == "imagenet":
            self.aug = transforms.Compose([
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                augmentation.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=63 / 255),
            ])
        else:
            self.aug = transforms.Compose([
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                augmentation.RandomHorizontalFlip(),
            ])

        self.mean = torch.tensor([0.5, 0.5, 0.5], requires_grad=True, device=self.device)
        self.std = torch.tensor([0.2, 0.2, 0.2], requires_grad=True, device=self.device)

        self.bn_mmt = bn_mmt
        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append(DeepInversionHook(m, self.bn_mmt))

    def synthesize(self, _cur_task=0):
        self._cur_task = _cur_task
        self.ep += 1
        self.student.eval()
        self.teacher.eval()
        best_cost = 1e6
        best_oh = 1e6

        # if self.ep % 200 == 0:
        #     self.generator = self.generator.reinit()

        best_inputs = None
        self.generator.re_init_le()

        targets, ys = self.generate_ys(cr=0.0)
        ys = ys.to(self.device)
        targets = targets.to(self.device)

        optimizer = torch.optim.Adam([
            {'params': self.generator.parameters()},
            {'params': [self.mean], 'lr': 0.01},
            {'params': [self.std], 'lr': 0.01}
        ], lr=self.lr_g, betas=[0.5, 0.999])

        for it in range(self.iterations):
            inputs = self.generator(targets=targets)
            inputs_aug = self.aug(inputs)
            inputs_aug = (inputs_aug - self.mean[None, :, None, None]) / (self.std[None, :, None, None])
            output_list = self.teacher(inputs_aug)
            t_out = output_list["logits"]
            feature = output_list["att"]

            loss_bn = sum([h.r_feature for h in self.hooks])
            loss_oh = custom_cross_entropy(t_out, ys.detach())

            if self.adv > 0 and (self.ep - 1 > self.ep_start):
                s_out = self.student(inputs_aug)["logits"]
                mask = (s_out.max(1)[1] == t_out.max(1)[1]).float()
                loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(
                    1) * mask).mean()  # decision adversarial distillation
            else:
                loss_adv = loss_oh.new_zeros(1)
            target_f = self.label_emb[targets]
            loss_f = torch.relu(torch.nn.functional.mse_loss(feature, target_f.detach()) - self.r)

            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv + self.ltc * loss_f

            if loss_oh.item() < best_oh:
                best_oh = loss_oh
            # print("%s - bn %s - bn %s - oh %s - adv %s -fr %s - %s - %s" % (
                # it,
                # float((loss_bn * self.bn).data.cpu().detach().numpy()),
                # float(loss_bn.data.cpu().detach().numpy()),
                # float((loss_oh).data.cpu().detach().numpy()),
                # float((loss_adv).data.cpu().detach().numpy()),
                # float(loss_f.data.cpu().detach().numpy()),
                # str(self.mean.detach().cpu()[0]),
                # str(self.std.detach().cpu()[0])))

            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if self.bn_mmt != 0:
            for h in self.hooks:
                h.update_mmt()

        if self.args['warmup'] <= self.ep:
            self.data_pool.add(best_inputs)
            self.student_train(self.student, self.teacher)

    def student_train(self, student, teacher):
        optimizer = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0002)
        student.train()
        teacher.eval()
        loader = self.get_all_syn_data()
        data_iter = DataIter(loader)
        criterion = KLDiv(T=T)

        prog_bar = tqdm(range(student_train_step))
        for _, com in enumerate(prog_bar):
            images = data_iter.next()
            images = images.cuda()
            with torch.no_grad():
                t_out_list = teacher(images)
                t_out = t_out_list["logits"]
                t_f = t_out_list["att"]
            s_out_list = student(images.detach())
            s_out = s_out_list["logits"]
            s_f = s_out_list["att"]
            loss_s = criterion(s_out, t_out.detach())
            s_loss_f = torch.nn.functional.mse_loss(s_f, t_f.detach())

            loss = loss_s + self.ltc*s_loss_f

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def get_all_syn_data(self):
        syn_dataset = UnlabeledImageDataset(self.save_dir, transform=self.transform, nums=self.args['nums'])
        loader = torch.utils.data.DataLoader(
            syn_dataset, batch_size=self.args["local_bs"], shuffle=True, persistent_workers=True,
            num_workers=self.args["num_worker"])
        return loader

    def generate_ys(self, cr=0.0):
        s = self.synthesis_batch_size // self.num_classes
        v = self.synthesis_batch_size % self.num_classes
        target = torch.randint(self.num_classes, (v,))

        for i in range(s):
            tmp_label = torch.tensor(range(0, self.num_classes))
            target = torch.cat((tmp_label, target))

        ys = torch.zeros(self.synthesis_batch_size, self.num_classes)
        ys.fill_(cr / (self.num_classes - 1))
        ys.scatter_(1, target.data.unsqueeze(1), (1 - cr))

        return target, ys

    def get_student(self):
        return self.student

    def get_teacher(self):
        return self.teacher
    

class LANDER(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientLANDER)
        self.old_network = self.global_model.copy().freeze()
        self.transform, self.normalizer = get_norm_and_transform(self.args.dataset)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def data_generation(self):
        if self.args.dataset == 'CIFAR100':
            img_size = 32
            img_shape = (3, 32, 32)
            generator = NLGenerator(ngf=64, img_size=img_size, nc=3, nl=10,
                                    label_emb=self.label_emb, le_emb_size=self.args['nz'],
                                    sbz=self.args['synthesis_batch_size'])
        elif self.args.dataset == 'IMAGENET1k':
            img_size = 224
            img_shape = (3, 224, 224)
            generator = NLGenerator_IN(ngf=64, img_size=img_size, nc=3, nl=10,
                                      label_emb=self.label_emb, le_emb_size=self.args['nz'],
                                      sbz=self.args['synthesis_batch_size'])
            
        student = copy.deepcopy(self.global_model)
        student.apply(weight_init)
        tmp_dir = os.path.join(self.save_dir, "task_{}".format(self.current_task))
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        synthesizer = NAYER(copy.deepcopy(self.global_model), student, generator, num_classes=self.clients[0].available_labels,
                            img_size=img_shape, save_dir=tmp_dir, transform=self.transform, normalizer=self.normalizer,
                            synthesis_batch_size=self.args['synthesis_batch_size'], iterations=self.args['g_steps'],
                            warmup=self.args['warmup'], lr_g=self.args['lr_g'], adv=self.args['adv'], bn=self.args['bn'],
                            oh=self.args['oh'], ltc=self.ltc, r=self.r, device=self.args["gpu"], bn_mmt=bn_mmt,
                            args=self.args, label_emb=self.label_emb)
            

    def train(self):

        for task in range(self.N_TASKS):

            print(f"\n================ Current Task: {task} =================")
            if task == 0:
                 # update labels info. for the first task
                available_labels = set()
                available_labels_current = set()
                available_labels_past = set()
                
                for u in self.clients:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(set(u.current_labels))

                for i, u in enumerate(self.clients):
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)
                    # print("task 0")
                    # print("client", i)
                    # print("available_label", u.available_labels)
                    # print("available_label_current", u.available_labels_current)
                    # print("available_label_past", u.available_labels_past)

            else:
                self.current_task = task
                
                torch.cuda.empty_cache()
                for i in range(len(self.clients)):
                    
                    if self.args.dataset == 'IMAGENET1k':
                        train_data, test_data, label_info = read_client_data_FCL_imagenet1k(i, task=task, classes_per_task=2, count_labels=True)
                    elif self.args.dataset == 'CIFAR100':
                        train_data, test_data, label_info = read_client_data_FCL_cifar100(i, task=task, classes_per_task=2, count_labels=True)
                    else:
                        raise NotImplementedError("Not supported dataset")

                    # update dataset
                    self.clients[i].next_task(train_data, test_data, label_info) # assign dataloader for new data
                    # print(self.clients[i].task_dict)

                # update labels info.
                available_labels = set()
                available_labels_current = set()
                available_labels_past = self.clients[0].available_labels

                for u in self.clients:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(set(u.current_labels))

                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)

                    # print(available_labels)

            # ============ train ==============
            self.global_model.update_fc(self.clients[0].available_labels)
            self.best_model = None
            self.lowest_loss = np.inf
            optimizer = torch.optim.SGD(self.global_model.parameters(), lr=self.args['local_lr'], momentum=0.9, weight_decay=self.args['weight_decay'])
            if self.args['dataset'] == "IMANGENET1k":
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args["global_rounds"], eta_min=1e-3)

            if self.current_task + 1 != self.N_TASKS:
                self.data_generation()

            for i in range(self.global_rounds):
                local_weights = []
                loss_weight = []
                glob_iter = i + self.global_rounds * task
                s_t = time.time()

                self.selected_clients = self.select_clients()
                self.send_models()

                if i%self.eval_gap == 0:
                    print(f"\n-------------Round number: {i}-------------")
                    self.eval(task=task, glob_iter=glob_iter, flag="global")

                for idx, client in enumerate(self.select_clients):
                    if task == 0 :
                        w, total_loss = client._local_update(copy.deepcopy(self.global_model))
                    else:
                        w, total_syn, total_local, total_loss = client._local_finetune(self.old_network,
                                                                                        copy.deepcopy(self.global_model),
                                                                                        self.current_task, 
                                                                                        scheduler.get_last_lr()[0])
                    
                    local_weights.append(copy.deepcopy(w))
                    loss_weight.append(total_loss)
                    del w
                    torch.cuda.empty_cache()
                scheduler.step()
                sum_loss = sum(loss_weight)
                if sum_loss < self.lowest_loss:
                    self.lowest_loss = sum_loss
                    self.best_model = copy.deepcopy(self.global_model.state_dict())

                self.receive_models()
                self.aggregate_parameters()

                if i%self.eval_gap == 0:
                    self.eval(task=task, glob_iter=glob_iter, flag="local")

                self.Budget.append(time.time() - s_t)

            self.eval_task(task=task, glob_iter=glob_iter, flag="local")
            
            # need eval before data update
            self.send_models()
            self.eval_task(task=task, glob_iter=glob_iter, flag="global")





        