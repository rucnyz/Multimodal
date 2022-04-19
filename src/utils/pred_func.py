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
    matrix = torch.mm(lsd2, (lsd1.T))  # (400,128)*(128,1600) 验证集第i个样本和训练集第j个样本的点积（训练集的label已知）
    label_num = label_onehot.sum(0, keepdim = True)
    # should sub 1.Avoid numerical errors; the number of samples of per label
    label_onehot = label_onehot.float()
    predicted_full_values = torch.mm(matrix, label_onehot) / label_num
    idx = torch.Tensor([1] * len(predicted_full_values)).long().view(-1, 1)  # 取第二列
    prob = predicted_full_values.gather(1, idx).reshape(1, -1) / torch.sum(predicted_full_values, dim=1)  # pos的概率：通过比例计算
    predicted = torch.max(predicted_full_values, dim = 1)[1]
    return predicted, prob[0]
