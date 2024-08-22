from src.models.cifar_alexnet import CIFAR10_AlexNet
from src.models.mnist_cnn import Mnist_CNN
from src.utils.torch_utils import setup_seed
import torch
import torch.nn as nn
import numpy as np


def choose_model(options):
    model_name = str(options['model_name']).lower()
    torch.manual_seed(2001)
    if model_name == 'mnist_cnn':

        for name, param in Mnist_CNN().named_parameters():
            if param.requires_grad:
                print(name, param.data)
                break
        return Mnist_CNN()
    if model_name == 'alex':
        return CIFAR10_AlexNet()



