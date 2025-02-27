import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.fedweit_utils import *

class NetModule:
    """ This module manages model networks and parameters
    Saves and loads all states whenever client is switched.
    Created by:
        Wonyong Jeong (wyjeong@kaist.ac.kr)
    """
    def __init__(self, args):
        self.args = args
        self.initializer = torch.nn.init.kaiming_normal_

        self.state = {}
        self.models = []
        self.heads = []
        self.decomposed_layers = {}
        self.initial_body_weights = []
        self.initial_heads_weights = []

        self.lid = 0
        self.adaptive_factor = 3
        self.input_shape = (3, 32, 32)
        
        if self.args.base_network == 'lenet':
            self.shapes = [
                (20, 3, 5, 5),
                (50, 20, 5, 5),
                (3200, 800),
                (800, 500)]
        
        if self.args.model in ['fedweit']:
            self.decomposed_variables = {
                'shared': [],
                'adaptive':{},
                'mask':{},
                'bias':{},
            }
            if self.args.model == 'fedweit':
                self.decomposed_variables['atten'] = {}
                self.decomposed_variables['from_kb'] = {}

    def init_state(self, cid):
        if self.args.model in ['fedweit']:
            self.state = {
                'client_id':  cid,
                'decomposed_weights': {
                    'shared': [],
                    'adaptive':{},
                    'mask':{},
                    'bias':{},
                },
                'heads_weights': self.initial_heads_weights,
            }
            if self.args.model == 'fedweit':
                self.state['decomposed_weights']['atten'] = {}
                self.state['decomposed_weights']['from_kb'] = {}
        else:
            self.state = {
                'client_id':  cid,
                'body_weights': self.initial_body_weights,
                'heads_weights': self.initial_heads_weights,
            } 

    def save_state(self):
        self.state['heads_weights'] = []
        for h in self.heads:
            self.state['heads_weights'].append(h.state_dict())
        if self.args.model in ['fedweit']:
            for var_type, layers in self.decomposed_variables.items():
                self.state['decomposed_weights'] = {
                    'shared': [layer.detach().cpu().numpy() for layer in self.decomposed_variables['shared']],
                    'adaptive':{tid: [layer.detach().cpu().numpy() for lid, layer in self.decomposed_variables['adaptive'][tid].items()] for tid in self.decomposed_variables['adaptive'].keys()},
                    'mask':{tid: [layer.detach().cpu().numpy() for lid, layer in self.decomposed_variables['mask'][tid].items()] for tid in self.decomposed_variables['mask'].keys()},
                    'bias':{tid: [layer.detach().cpu().numpy() for lid, layer in self.decomposed_variables['bias'][tid].items()] for tid in self.decomposed_variables['bias'].keys()},
                }
                if self.args.model == 'fedweit':
                    self.state['decomposed_weights']['from_kb'] = {tid: [layer.detach().cpu().numpy() for lid, layer in self.decomposed_variables['from_kb'][tid].items()] for tid in self.decomposed_variables['from_kb'].keys()}
                    self.state['decomposed_weights']['atten'] = {tid: [layer.detach().cpu().numpy() for lid, layer in self.decomposed_variables['atten'][tid].items()] for tid in self.decomposed_variables['atten'].keys()}
        else:
            self.state['body_weights'] = self.model_body.state_dict()
        
        np_save(self.args.state_dir, '{}_net.npy'.format(self.state['client_id']), self.state)

    def load_state(self, cid):
        self.state = np_load(os.path.join(self.args.state_dir, '{}_net.npy'.format(cid))).item()

        for i, h in enumerate(self.state['heads_weights']):
            self.heads[i].load_state_dict(h)

        if self.args.model in ['fedweit']:
            for var_type, values in self.state['decomposed_weights'].items():
                if var_type == 'shared':
                    for lid, weights in enumerate(values):
                        self.decomposed_variables['shared'][lid].data = torch.tensor(weights)
                else:
                    for tid, layers in values.items():
                        for lid, weights in enumerate(layers):    
                            self.decomposed_variables[var_type][tid][lid].data = torch.tensor(weights)
        else:
            self.model_body.load_state_dict(self.state['body_weights'])

    def init_global_weights(self):
        if self.args.model in ['fedweit']:
            global_weights = []
            for i in range(len(self.shapes)):
                global_weights.append(self.initializer(torch.empty(self.shapes[i])).numpy())
        else:
            if self.args.base_network == 'lenet':
                body = self.build_lenet_body(decomposed=False)
            global_weights = body.state_dict()
        return global_weights

    def init_decomposed_variables(self, initial_weights):
        self.decomposed_variables['shared'] = [torch.nn.Parameter(torch.tensor(initial_weights[i]), requires_grad=True) for i in range(len(self.shapes))]
        for tid in range(self.args.num_tasks):
            for lid in range(len(self.shapes)):
                var_types = ['adaptive', 'bias', 'mask'] if self.args.model == 'apd' else ['adaptive', 'bias', 'mask', 'atten', 'from_kb']
                for var_type in var_types:
                    self.create_variable(var_type, lid, tid)

    def create_variable(self, var_type, lid, tid=None):
        trainable = True 
        if tid not in self.decomposed_variables[var_type]:
            self.decomposed_variables[var_type][tid] = {}
        if var_type == 'adaptive':
            init_value = self.decomposed_variables['shared'][lid].detach().cpu().numpy()/self.adaptive_factor
        elif var_type == 'atten':
            shape = (int(round(self.args.num_clients*self.args.frac_clients)),)
            if tid == 0:
                trainable = False
                init_value = np.zeros(shape).astype(np.float32)
            else:
                init_value = self.initializer(torch.empty(shape)).numpy()
        elif var_type == 'from_kb':
            shape = np.concatenate([self.shapes[lid], [int(round(self.args.num_clients*self.args.frac_clients))]], axis=0)
            trainable = False
            if tid == 0:
                init_value = np.zeros(shape).astype(np.float32)
            else:
                init_value = self.initializer(torch.empty(shape)).numpy()
        else:
            init_value = self.initializer(torch.empty(self.shapes[lid][-1],)).numpy()
        var = torch.nn.Parameter(torch.tensor(init_value), requires_grad=trainable)
        self.decomposed_variables[var_type][tid][lid] = var

    def get_variable(self, var_type, lid, tid=None):
        if var_type == 'shared':
            return self.decomposed_variables[var_type][lid]
        else:
            return self.decomposed_variables[var_type][tid][lid]

    def generate_mask(self, mask):
        return torch.sigmoid(mask)

    def get_model_by_tid(self, tid):
        if self.args.model in ['fedweit']:
            self.switch_model_params(tid)
        return self.models[tid]

    def get_trainable_variables(self, curr_task, head=True):
        if self.args.model in ['fedweit']:
            return self.get_decomposed_trainable_variables(curr_task, retroactive=False, head=head)
        else:
            if head:
                return self.models[curr_task].parameters()
            else:
                return self.model_body.parameters()

    def get_decomposed_trainable_variables(self, curr_task, retroactive=False, head=True):
        prev_variables = ['mask', 'bias', 'adaptive'] if self.args.model == 'apd' else ['mask', 'bias', 'adaptive', 'atten']
        trainable_variables = [sw for sw in self.decomposed_variables['shared']]
        if retroactive:
            for tid in range(curr_task+1):
                for lid in range(len(self.shapes)):
                    for pvar in prev_variables:
                        if pvar == 'bias' and tid < curr_task:
                            continue
                        if pvar == 'atten' and tid == 0:
                            continue
                        trainable_variables.append(self.get_variable(pvar, lid, tid))
        else:
            for lid in range(len(self.shapes)):
                for pvar in prev_variables:
                    if pvar == 'atten' and curr_task == 0:
                        continue
                    trainable_variables.append(self.get_variable(pvar, lid, curr_task))
        if head:
            head = self.heads[curr_task]
            trainable_variables.append(head.weight)
            trainable_variables.append(head.bias)
        return trainable_variables

    def get_body_weights(self, task_id=None):
        if self.args.model in ['fedweit']:
            prev_weights = {}
            for lid in range(len(self.shapes)):
                prev_weights[lid] = {}
                sw = self.get_variable(var_type='shared', lid=lid).detach().cpu().numpy()
                for tid in range(task_id):
                    prev_aw = self.get_variable(var_type='adaptive', lid=lid, tid=tid).detach().cpu().numpy()
                    prev_mask = self.get_variable(var_type='mask', lid=lid, tid=tid).detach().cpu().numpy()
                    prev_mask_sig = self.generate_mask(torch.tensor(prev_mask)).detach().cpu().numpy()
                    #################################################
                    prev_weights[lid][tid] = sw * prev_mask_sig + prev_aw
                    #################################################
            return prev_weights
        else:
            return self.model_body.state_dict()

    def set_body_weights(self, body_weights):
        if self.args.model in ['fedweit']:
            for lid, wgt in enumerate(body_weights):
                sw = self.get_variable('shared', lid)
                sw.data = torch.tensor(wgt)
        else:
            self.model_body.load_state_dict(body_weights)
    
    def switch_model_params(self, tid):
        for lid, dlay in self.decomposed_layers.items():
            dlay.sw = self.get_variable('shared', lid)
            dlay.aw = self.get_variable('adaptive', lid, tid)
            dlay.bias = self.get_variable('bias', lid, tid)
            dlay.mask = self.generate_mask(self.get_variable('mask', lid, tid))
            if self.args.model == 'fedweit':
                dlay.atten = self.get_variable('atten', lid, tid) 
                dlay.aw_kb = self.get_variable('from_kb', lid, tid) 

    def add_head(self, body):
        head = nn.Linear(body.output_shape[1], self.args.num_classes)
        self.heads.append(head)
        self.initial_heads_weights.append(head.state_dict())
        return nn.Sequential(body, head) # multiheaded model

    def build_lenet(self, initial_weights, decomposed=False):
        self.models = []
        self.model_body = self.build_lenet_body(initial_weights, decomposed=decomposed)
        self.set_body_weights(initial_weights)
        self.initial_body_weights = initial_weights
        for i in range(self.args.num_tasks):
            self.models.append(self.add_head(self.model_body))

    def build_lenet_body(self, initial_weights=None, decomposed=False):
        if decomposed:
            self.init_decomposed_variables(initial_weights)
            tid = 0
            layers = []
            for lid in [0, 1]:
                self.decomposed_layers[self.lid] = self.conv_decomposed(lid, tid,
                    filters = self.shapes[lid][-1],
                    kernel_size = (self.shapes[lid][0], self.shapes[lid][1]),
                    strides = (1,1),
                    padding = 'same',
                    acti = 'relu')
                layers.append(self.decomposed_layers[self.lid])
                self.lid += 1
                layers.append(nn.LocalResponseNorm(4, alpha=0.001/9.0, beta=0.75, k=1.0))
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            layers.append(nn.Flatten())
            for lid in [2, 3]:
                self.decomposed_layers[self.lid] = self.dense_decomposed(
                    lid, tid,
                    units=self.shapes[lid][-1],
                    acti='relu')
            layers.append(self.decomposed_layers[self.lid])
            self.lid += 1
            model = nn.Sequential(*layers)
        else:
            layers = []
            layers.append(nn.Conv2d(3, 20, kernel_size=5, padding=2))
            layers.append(nn.LocalResponseNorm(4, alpha=0.001/9.0, beta=0.75, k=1.0))
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            layers.append(nn.Conv2d(20, 50, kernel_size=5, padding=2))
            layers.append(nn.LocalResponseNorm(4, alpha=0.001/9.0, beta=0.75, k=1.0))
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            layers.append(nn.Flatten())
            layers.append(nn.Linear(3200, 800))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(800, 500))
            layers.append(nn.ReLU())
            model = nn.Sequential(*layers)
        return model

    def conv_decomposed(self, lid, tid, filters, kernel_size, strides, padding, acti):
        return  DecomposedConv(
            name        = 'layer_{}'.format(lid),
            filters     = filters,
            kernel_size = kernel_size,
            strides     = strides,
            padding     = padding,
            activation  = acti,
            lambda_l1   = self.args.lambda_l1,
            lambda_mask = self.args.lambda_mask,
            shared      = self.get_variable('shared', lid),
            adaptive    = self.get_variable('adaptive', lid, tid),
            from_kb     = self.get_variable('from_kb', lid, tid),
            atten       = self.get_variable('atten', lid, tid),
            bias        = self.get_variable('bias', lid, tid), use_bias=True,
            mask        = self.generate_mask(self.get_variable('mask', lid, tid)),
            kernel_regularizer = nn.L2(self.args.wd))

    def dense_decomposed(self, lid, tid, units, acti):
        return DecomposedDense(
            name        = 'layer_{}'.format(lid),
            activation  = acti,
            units       = units,
            lambda_l1   = self.args.lambda_l1,
            lambda_mask = self.args.lambda_mask,
            shared      = self.get_variable('shared', lid),
            adaptive    = self.get_variable('adaptive', lid, tid),
            from_kb     = self.get_variable('from_kb', lid, tid),
            atten       = self.get_variable('atten', lid, tid),
            bias        = self.get_variable('bias', lid, tid), use_bias=True,
            mask        = self.generate_mask(self.get_variable('mask', lid, tid)),
            kernel_regularizer = nn.L2(self.args.wd))

# Layers
class DecomposedDense(nn.Module):
    """ Custom dense layer that decomposes parameters into shared and specific parameters.
    """
    def __init__(self, 
                 units,
                 activation=None,
                 use_bias=False,
                 lambda_l1=None,
                 lambda_mask=None,
                 shared=None,
                 adaptive=None,
                 from_kb=None,
                 atten=None,
                 mask=None,
                 bias=None):
        super(DecomposedDense, self).__init__()
        
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        
        self.sw = shared
        self.aw = adaptive
        self.mask = mask
        self.bias = bias
        self.aw_kb = from_kb
        self.atten = atten
        self.lambda_l1 = lambda_l1
        self.lambda_mask = lambda_mask
    
    def l1_pruning(self, weights, hyp):
        hard_threshold = (weights.abs() > hyp).float()
        return weights * hard_threshold
    
    def forward(self, inputs):
        aw = self.aw if self.training else self.l1_pruning(self.aw, self.lambda_l1)
        mask = self.mask if self.training else self.l1_pruning(self.mask, self.lambda_mask)
        atten = self.atten
        aw_kbs = self.aw_kb
        
        self.my_theta = self.sw * mask + aw + torch.sum(aw_kbs * atten, dim=-1)
        
        outputs = torch.matmul(inputs, self.my_theta)
        
        if self.use_bias:
            outputs = outputs + self.bias
        
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

class DecomposedConv(nn.Module):
    """ Custom conv layer that decomposes parameters into shared and specific parameters.
    """
    def __init__(self, 
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 activation=None,
                 use_bias=False,
                 lambda_l1=None,
                 lambda_mask=None,
                 shared=None,
                 adaptive=None,
                 from_kb=None,
                 atten=None,
                 mask=None,
                 bias=None):
        super(DecomposedConv, self).__init__()
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.use_bias = use_bias
        
        self.sw = shared
        self.aw = adaptive
        self.mask = mask
        self.bias = bias
        self.aw_kb = from_kb
        self.atten = atten
        self.lambda_l1 = lambda_l1
        self.lambda_mask = lambda_mask
    
    def l1_pruning(self, weights, hyp):
        hard_threshold = (weights.abs() > hyp).float()
        return weights * hard_threshold
    
    def forward(self, inputs):
        aw = self.aw if self.training else self.l1_pruning(self.aw, self.lambda_l1)
        mask = self.mask if self.training else self.l1_pruning(self.mask, self.lambda_mask)
        atten = self.atten
        aw_kbs = self.aw_kb
        
        self.my_theta = self.sw * mask + aw + torch.sum(aw_kbs * atten, dim=-1)
        
        outputs = F.conv2d(inputs, self.my_theta, stride=self.strides, padding=self.padding, dilation=self.dilation_rate)
        
        if self.use_bias:
            outputs = outputs + self.bias.view(1, -1, 1, 1)
        
        if self.activation is not None:
            return self.activation(outputs)
        return outputs
