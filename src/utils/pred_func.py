import numpy as np
import torch


def amax(x):
    return np.argmax(x, axis = 1)


def multi_label(x):
    return (x > 0)


def accuracy_count(predicted, y):
    return (predicted == y).sum().item()


def ave(lsd1, lsd2, label_onehot):
    """In most cases, this method is used to predict the highest accuracy.
    :param lsd1: train set's latent space data
    :param lsd2: test set's latent space data
    :param label: label of train set
    :return: Predicted label
    """
    matrix = torch.mm(lsd2, (lsd1.T))
    label_num = label_onehot.sum(0, keepdim = True)
    # should sub 1.Avoid numerical errors; the number of samples of per label
    label_onehot = label_onehot.float()
    predicted_full_values = torch.mm(matrix, label_onehot) / label_num
    predicted = torch.max(predicted_full_values, dim = 1)[1]
    return predicted
