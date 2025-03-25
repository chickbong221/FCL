import copy
from torch.utils.data import DataLoader
import torch
import glog as logger
import numpy as np
import wandb

from flcore.clients.clientbase import Client
from utils.precise_utils import str_in_list, Meter

eps = 1e-30

class ClientPreciseFCL(Client):
    def __init__(self, args, id, train_data, test_data, classifier_head_list=[], **kwargs):
        super().__init__(args, id, train_data, test_data, **kwargs)
        
        self.args = args
        self.k_loss_flow = args.k_loss_flow
        self.classifier_head_list = classifier_head_list
        self.use_lastflow_x = args.use_lastflow_x
    
    def train(
        self,
        glob_iter,
        global_classifier,
        verbose
    ):
        '''
        @ glob_iter: the overall iterations across all tasks
        
        '''
        trainloader = self.load_train_data()
        correct = 0
        sample_num = 0
        cls_meter = Meter()
        for iteration in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                last_classifier = None
                last_flow = None
                if type(self.last_copy)!=type(None):
                    last_classifier = self.last_copy.classifier
                    last_classifier.eval()
                    if self.algorithm=='PreciseFCL':
                        last_flow = self.last_copy.flow
                        last_flow.eval()

                if self.algorithm=='PreciseFCL' and self.k_loss_flow>0:
                    self.model.classifier.eval()
                    self.model.flow.train()
                    flow_result = self.model.train_a_batch(
                        x, y, train_flow=True, flow=None, last_flow=last_flow,
                        last_classifier = last_classifier,
                        global_classifier = global_classifier,
                        classes_so_far = self.classes_so_far,
                        classes_past_task = self.classes_past_task,
                        available_labels = self.available_labels,
                        available_labels_past = self.available_labels_past)
                    cls_meter._update(flow_result, batch_size=x.shape[0])

                flow = None
                if self.algorithm=='PreciseFCL':
                    if self.use_lastflow_x:
                        flow = last_flow
                    else:
                        flow = self.model.flow
                        flow.eval()

                self.model.classifier.train()
                cls_result = self.model.train_a_batch(
                    x, y, train_flow=False, flow=flow, last_flow=last_flow,
                    last_classifier = last_classifier,
                    global_classifier = global_classifier,
                    classes_so_far = self.classes_so_far,
                    classes_past_task = self.classes_past_task,
                    available_labels = self.available_labels,
                    available_labels_past = self.available_labels_past)

                #c_loss_all += result['c_loss']
                correct += cls_result['correct']
                sample_num += x.shape[0]
                cls_meter._update(cls_result, batch_size=x.shape[0])

        acc = float(correct)/sample_num
        result_dict = cls_meter.get_scalar_dict('global_avg')
        if 'flow_loss' not in result_dict.keys():
            result_dict['flow_loss'] = 0
        if 'flow_loss_last' not in result_dict.keys():
            result_dict['flow_loss_last'] = 0

        if verbose:
            logger.info(("Training for user {:d}; Acc: {:.2f} %%; c_loss: {:.4f}; kd_loss: {:.4f}; flow_prob_mean: {:.4f}; "
                         "flow_loss: {:.4f}; flow_loss_last: {:.4f}; c_loss_flow: {:.4f}; kd_loss_flow: {:.4f}; "
                         "kd_loss_feature: {:.4f}; kd_loss_output: {:.4f}").format(
                                        self.id, acc*100.0, result_dict['c_loss'], result_dict['kd_loss'],
                                        result_dict['flow_prob_mean'], result_dict['flow_loss'], result_dict['flow_loss_last'],
                                        result_dict['c_loss_flow'], result_dict['kd_loss_flow'],
                                        result_dict['kd_loss_feature'], result_dict['kd_loss_output']))

        return {'acc': acc, 'c_loss': result_dict['c_loss'], 'kd_loss': result_dict['kd_loss'], 'flow_prob_mean': result_dict['flow_prob_mean'],
                 'flow_loss': result_dict['flow_loss'], 'flow_loss_last': result_dict['flow_loss_last'], 'c_loss_flow': result_dict['c_loss_flow'],
                   'kd_loss_flow': result_dict['kd_loss_flow']}
