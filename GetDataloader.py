'''
coding:utf-8
@Software:PyCharm
@Time:2024/2/29 10:25
@Author:chenGuiBin
@description:
'''
import itertools

import scipy.io
import numpy as np
import torch
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import transforms

def standardize_data(data):
    """
    标准化处理函数
    Args:
        data: 输入的数据张量

    Returns:
        标准化后的数据张量
    """
    mean = torch.mean(data, dim=0)
    std = torch.std(data, dim=0)
    return (data - mean) / std


def add_noise(data, noise_level=0.01):
    noise = torch.randn(data.size()) * noise_level
    noisy_data = data + noise
    return noisy_data

# 划分为训练集、测试集和验证集,留一法
def load_data(data_name, train_name_list, val_name, batch_size):
    # 加载.mat文件
    struct_data = scipy.io.loadmat(data_name)

    # 处理训练数据
    # 初始化空列表，用于存储所有的data和label
    train_datas = []
    train_labels = []
    # 遍历训练数据的每个结构体数据
    for train_name in train_name_list:
        # 获取当前结构体中的data和label
        train_data = struct_data[train_name]['Data'][0, 0]  # 假设data存储在结构体的'data'字段中
        train_label = struct_data[train_name]['Label'][0, 0]  # 假设label存储在结构体的'label'字段中

        # 将data和label添加到列表中
        train_datas.append(train_data)
        train_labels.append(train_label)
    # 将列表转换为数组，并沿着指定维度拼接
    train_concatenated_datas = np.concatenate(train_datas, axis=0)
    train_concatenated_labels = np.concatenate(train_labels, axis=0)
    # 将其转为tensor张量，加载为DataLoader
    # 将列表转换为张量
    train_data_tensor = torch.tensor(train_concatenated_datas, dtype=torch.float32)
    train_label_tensor = torch.tensor(train_concatenated_labels, dtype=torch.int64)
    # 标准化训练数据
    # train_data_tensor = standardize_data(train_data_tensor)
    # 添加噪声
    train_data_tensor = add_noise(train_data_tensor)
    # 将数据和标签组合成数据集
    train_dataset = TensorDataset(train_data_tensor, train_label_tensor)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 处理测试数据
    # struct_data_val = scipy.io.loadmat('data/DataSet5SPre.mat')
    # val_data = struct_data_val[val_name]['Data'][0, 0]  # 假设data存储在结构体的'data'字段中
    # val_label = struct_data_val[val_name]['Label'][0, 0]  # 假设label存储在结构体的'label'字段中


    val_data = struct_data[val_name]['Data'][0, 0]  # 假设data存储在结构体的'data'字段中
    val_label = struct_data[val_name]['Label'][0, 0]  # 假设label存储在结构体的'label'字段中
    # 将其转为tensor张量，加载为DataLoader
    # 将列表转换为张量
    val_data_tensor = torch.tensor(val_data, dtype=torch.float32)
    val_label_tensor = torch.tensor(val_label, dtype=torch.int64)
    # 标准化训练数据
    # val_data_tensor = standardize_data(val_data_tensor)
    # 将数据和标签组合成数据集
    val_dataset = TensorDataset(val_data_tensor, val_label_tensor)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_data_loader, val_data_loader

