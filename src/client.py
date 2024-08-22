from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import numpy as np
from src.utils.flops_counter import get_model_complexity_info
from src.utils.torch_utils import get_flat_grad, get_state_dict, get_flat_params_from, set_flat_params_to
import torch.nn as nn
import torch
import copy
criterion = F.cross_entropy
mse_loss = nn.MSELoss()


class Client():
    def __init__(self, options, id, attr, local_dataset, model, optimizer, ):
        self.options = options
        self.id = id
        self.local_dataset = local_dataset
        self.model = model
        self.gpu = options['gpu']
        self.optimizer = optimizer
        self.flops, self.params_num, self.model_bytes = \
            get_model_complexity_info(self.model, 784, gpu=options['gpu'])
        self.attr_dict = attr.get_client_attr(self.id)
        # self.cpu_fr =
    def get_client_specific_attribute(self,):
        return self.attr_dict.get_client_attr(self.id)

    def get_model_parameters(self):
        state_dict = self.model.state_dict()
        return state_dict

    def set_model_parameters(self, model_parameters_dict):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_parameters_dict[key]
        self.model.load_state_dict(state_dict)

    def local_train(self, ):
        bytes_w = self.model_bytes
        begin_time = time.time()
        local_model_paras, dict = self.local_update(self.local_dataset, self.options, )
        end_time = time.time()
        bytes_r = self.model_bytes
        stats = {'id': self.id, 'bytes_w': bytes_w, 'bytes_r': bytes_r,
                 "time": round(end_time - begin_time, 2)}
        stats.update(dict)
        return (len(self.local_dataset), local_model_paras), stats

    def local_update(self, local_dataset, options, ):
        localTrainDataLoader = DataLoader(local_dataset, batch_size=options['batch_size'], shuffle=True)
        self.model.train()
        # print(self.optimizer.param_groups[0]['lr'])
        train_loss = train_acc = train_total = 0
        for epoch in range(options['local_epoch']):
            train_loss = train_acc = train_total = 0
            for X, y in localTrainDataLoader:
                if self.gpu:
                    X, y = X.cuda(), y.cuda()
                pred = self.model(X)
                loss = criterion(pred, y)
                loss.backward()
                # print(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)
                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size
        local_model_paras = self.get_model_parameters()
        comp = self.options['local_epoch'] * train_total * self.flops
        return_dict = {"id": self.id,
                       "comp": comp,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total}

        return local_model_paras, return_dict

    def getLocalEngery(self):
        localEngery = (10 ** -26) * (self.attr_dict['cpu_frequency'] * 1000000000) ** 2 * 10000 * len(self.local_dataset)
        return localEngery

    def getUploadEngery(self):
        uploadEngery = self.attr_dict['transmit_power'] * self.getUploadDelay()
        return uploadEngery

    def getLocalDelay(self):
        localDelay = 10000 * len(self.local_dataset) * self.options['local_epoch'] / (self.attr_dict['cpu_frequency'] * 1000000000)
        return localDelay

    def getUploadDelay(self):
        R_K = self.attr_dict['B'] * 1000000 * np.log2(1 + self.attr_dict['transmit_power'] * self.attr_dict['g_N0']) # 1M bit / s / self.B
        uploadDelay = 6.35 / (R_K / 8 / 1024 / 1024) # 100KB 0.1M  # 1S
        return uploadDelay

    def getSumEngery(self):
        return self.getUploadEngery() + self.getLocalEngery()

    def getSumDelay(self):
        return self.getUploadDelay() + self.getLocalDelay()
