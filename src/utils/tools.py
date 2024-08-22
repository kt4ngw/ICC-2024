import numpy as np





import pickle
import json
import numpy as np
import os
import time
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset
from PIL import Image

import random
import numpy as np
import torch

class RandomSeedManager:
    def __init__(self, seed=None):
        self.seed = seed

    def set_seed(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def unset_seed(self):
        random.seed()
        np.random.seed()
        torch.manual_seed(np.random.randint(1, 10000))
        torch.cuda.manual_seed_all(np.random.randint(1, 10000))
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

class Metrics(object):
    def __init__(self, options, clients, name=''):
        self.options = options

        num_rounds = options['round_num'] + 1
        self.bytes_written = {c.id: [0] * num_rounds for c in clients}
        self.client_computations = {c.id: [0] * num_rounds for c in clients}
        self.bytes_read = {c.id: [0] * num_rounds for c in clients}

        # global_test_data
        self.loss_on_g_test_data = [0] * num_rounds
        self.acc_on_g_test_data = [0] * num_rounds


        # cost time and delay
        self.accumulation_delay = [0] * num_rounds
        self.accumulation_energy = [0] * num_rounds


        self.result_path = mkdir(os.path.join('./result', self.options['dataset_name'] + str(self.options['dirichlet'])))
        suffix = '{}_sd{}_lr{}_ne{}_bs{}'.format(name,
                                                    options['seed'],
                                                    options['lr'],
                                                    options['round_num'],
                                                    options['batch_size'],
                                                )
        # self.exp_name = '{}_{}_{}_{}'.format(time.strftime('%Y-%m-%dT%H-%M-%S'), options['algorithm'],
        #                                      options['model_name'], suffix)
        self.exp_name = '{}_{}_{}_{}'.format(options['algorithm'],
                                             options['model_name'], options['opti'], suffix)

        train_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'train.event'))
        test_event_folder = mkdir(os.path.join(self.result_path, self.exp_name, 'eval.event'))
        self.train_writer = SummaryWriter(train_event_folder)
        self.eval_writer = SummaryWriter(test_event_folder)

    def update_communication_stats(self, round_i, stats):
        id, bytes_w, comp, bytes_r = \
            stats['id'], stats['bytes_w'], stats['comp'], stats['bytes_r']
        self.bytes_written[id][round_i] += bytes_w
        self.client_computations[id][round_i] += comp
        self.bytes_read[id][round_i] += bytes_r

    def extend_communication_stats(self, round_i, stats_list):
        for stats in stats_list:
            self.update_communication_stats(round_i, stats)

    def update_test_stats(self, round_i, eval_stats):
        self.loss_on_g_test_data[round_i] = eval_stats['loss']
        self.acc_on_g_test_data[round_i] = eval_stats['acc']

        self.eval_writer.add_scalar('test_loss', eval_stats['loss'], round_i)
        self.eval_writer.add_scalar('test_acc', eval_stats['acc'], round_i)

    def update_cost(self, round_i, delay, energy):
        self.accumulation_delay[round_i] = delay
        self.accumulation_energy[round_i] = energy



    def write(self):
        metrics = dict()
        metrics['dataset'] = self.options['dataset_name']
        metrics['loss_on_g_test_data'] = self.loss_on_g_test_data
        metrics['acc_on_g_test_data'] = self.acc_on_g_test_data
        metrics['accumulation_delay'] = self.accumulation_delay
        metrics['accumulation_energy'] = self.accumulation_energy
        metrics_dir = os.path.join(self.result_path, self.exp_name, 'metrics.json')

        with open(metrics_dir, 'w') as ouf:
            json.dump(metrics, ouf)


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    np.random.seed(2001)
    n_classes = train_labels.max() + 1

    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]

    # 首先，为每个客户端分配一个样本
    for i in range(n_clients):
        random_class = np.random.choice(n_classes)
        random_sample = np.random.choice(class_idcs[random_class])
        client_idcs[i].append(random_sample)

    # 分配余下的样本给每个客户端
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(k_idcs, (np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int))):
            client_idcs[i] += idcs.tolist()

    client_idcs = [np.array(idcs) for idcs in client_idcs]
    # 假设 client_idcs 是一个包含多个子列表的列表
    result = []
    for i, idcs in enumerate(client_idcs):
        result.append(idcs.tolist())
    return client_idcs, result

import numpy as np
import matplotlib.pyplot as plt


def plot_client_class_categories(client_idcs, n_clients, n_classes, train_labels):
    client_class_counts = np.zeros((n_clients, n_classes))

    for i, idcs in enumerate(client_idcs):
        class_counts = np.bincount(train_labels[idcs], minlength=n_classes)
        client_class_counts[i, :] = class_counts

    fig, ax = plt.subplots(figsize=(16, 8))

    max_count = np.max(client_class_counts)
    color = '#D5BA82'  # Single color for all scatter points
    print(client_class_counts)
    for i, class_counts in enumerate(client_class_counts[:20]):
        for j, count in enumerate(class_counts[:20]):
            scatter_size = count / max_count * 1000  # Adjust scatter size based on count
            ax.scatter([i], [j], s=scatter_size, color=color, alpha=0.7)

    ax.set_xlabel('Client ID', fontsize=35)
    ax.set_ylabel('Class', fontsize=35)
    # ax.set_title('Classes Present in Each Client')
    ax.set_xticks(np.arange(n_clients))
    ax.set_xticklabels([f'{i}' for i in range(n_clients)], fontsize=35)
    ax.set_yticks(np.arange(n_classes))
    ax.set_yticklabels([f'{i}' for i in range(n_classes)], fontsize=35)

    plt.tight_layout()
    plt.show()
