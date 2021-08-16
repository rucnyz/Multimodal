import numpy as np
import torch


def amax(x):
    return np.argmax(x, axis = 1)


def multi_label(x):
    return (x > 0)


def accuracy_count(predicted, y):
    _, predicted = torch.max(predicted.data, 1)
    return (predicted == y).sum().item()
