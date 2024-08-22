import argparse
import torch
from getdata import GetDataSet
from src.utils.tools import dirichlet_split_noniid
from src.utils.paras_generate import paraGeneration
import importlib
from src.utils.torch_utils import setup_seed
from src.utils.tools import plot_client_class_categories
# GLOBAL PARAMETERS
DATASETS = ['mnist',  'cifar10']
TRAINERS = {'fedavg': 'FedAvgTrainer',
            'propose': 'ProposeTrainer',
            'propose2': 'Propose2Trainer',
            'nws': 'NumWSamplingTrainer',
            }
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

OPTIMIZERS = TRAINERS.keys()
def input_options():
    parser = argparse.ArgumentParser()
    # iid
    parser.add_argument( '-is_iid', type=bool, default=True, help='data distribution is iid.')
    parser.add_argument( '--dataset_name', type=str, default='mnist_dir_', help='name of dataset.')
    parser.add_argument('--model_name', type=str, default='mnist_cnn', help='the model to train')
    parser.add_argument('--gpu', type=bool, default=True, help='gpu id to use(e.g. 0,1,2,3)')
    parser.add_argument('--round_num', type=int, default=301, help='number of round in comm')
    parser.add_argument( '--num_of_clients', type=int, default=100, help='numer of the clients')
    parser.add_argument( '--c_fraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
    parser.add_argument('--local_epoch', type=int, default=5, help='local train epoch')
    parser.add_argument( '--batch_size', type=int, default=32, help='local train batch size')
    parser.add_argument( "--lr", type=float, default=0.1, help="learning rate, \
                        use value from origin paper as default")
    parser.add_argument( '--transmit_power', type=int, default=1, help='transmitpower')
    parser.add_argument( '--gn0', type=int, default=1, help='gno')
    parser.add_argument('--seed', help='seed for randomness;', type=int, default=0)
    parser.add_argument( '--weight_decay', help='weight_decay;', type=int, default=1)
    parser.add_argument( '--algorithm', help='algorithm;', choices=OPTIMIZERS, type=str, default='propose')
    parser.add_argument( '--dirichlet', help='Dirichlet;', type=float, default=0.1)
    parser.add_argument('--opti', help='Dirichlet;', type=str, default='sgd')
    args = parser.parse_args()
    options = args.__dict__
    dataset = GetDataSet(options['dataset_name'][:5]) # 拿到数据集 分配完再导入
    client_label, result = dirichlet_split_noniid(dataset.trainLabel, options['dirichlet'], options['num_of_clients'])
    # # 保存客户端标签到文本文件
    # # 将列表元素连接成一个字符串，以逗号分隔
    # list_str = ' '.join(map(str, dataset.trainLabel.tolist()))
    # with open('client_labels.txt', 'w') as file:
    #     for label in result[:20]:
    #         for i in label:
    #             file.write(f'{i} ')
    #         file.write(f'\n')
    #     for i in list_str:
    #         file.write(f'{i}')
    #     file.write(f'\n')
    plot_client_class_categories(client_label[:20], 20, 10, dataset.trainLabel)

    cpu_frequency, B, transmit_power, g_N0 = paraGeneration(options)
    setup_seed(options['seed'])
    trainer_path = 'src.trainers.%s' % options['algorithm']
    mod = importlib.import_module(trainer_path)
    trainer_class = getattr(mod, TRAINERS[options['algorithm']])
    trainer = trainer_class(options, dataset, client_label, cpu_frequency, B, transmit_power, g_N0)
    return trainer


