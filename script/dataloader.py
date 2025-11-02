import os
import numpy as np
import pandas as pd
import torch


def load_data(dataset_name, len_train, len_val):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    gt = pd.read_csv(os.path.join(dataset_path, '关键参数去掉启停过程20250106.csv'),encoding='gbk')
    train = gt[: len_train]
    val = gt[len_train: len_train + len_val]
    test = gt[len_train + len_val:]

    return train, val, test

def data_transform(data, n_his, n_pred, device):
    # produce data slices for x_data and y_data

    n_vertex = data.shape[1]
    len_record = len(data)
    num = len_record - n_his - n_pred
    
    x = np.zeros([num, 1, n_his, n_vertex])
    y = np.zeros([num, n_vertex])
    
    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data[head: tail].reshape(1, n_his, n_vertex)
        y[i] = data[tail + n_pred - 1]

    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)