# alt_model.py
# Copyright (c) 2023 Sirui Li (sirui.li@murdoch.edu.au) and Kevin Wong (K.Wong@murdoch.edu.au)
# ICT203 - Artificial Intelligence and Intelligent Agents
# Murdoch University

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class ALTModel(nn.Module):
    """
    A custom PyTorch model for your alternative neural network-based model.
    """
    def __init__(self):
        "*** YOUR CODE HERE ***"
        super(ALTModel, self).__init__()
        self.hidden1 = nn.Linear(784, 392)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        self.hidden2 = nn.Linear(392, 128)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        "*** YOUR CODE HERE ***"
        x = self.hidden1(x)
        x = self.act1(x)
        x = self.hidden2(x)
        x = self.act2(x)
        x = self.output(x)
        return x

