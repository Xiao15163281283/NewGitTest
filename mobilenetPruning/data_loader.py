import numpy as np

import torch
from torchvision import datasets
from torchvision import transforms

def get_train_loader(data_dir,
                     batch_size,
                     random_seed,
                     shuffle=True,
                     num_workers=4,
                     pin_memory=True):
    # define transforms
    trans = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 归一化
    ])

    # load dataset
    dataset = datasets.CIFAR10(root=data_dir,
                                transform=trans,
                                download=False,
                                train=True)
    if shuffle:
        np.random.seed(random_seed)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader



def get_test_loader(data_dir,
                    batch_size,
                    num_workers=4,
                    pin_memory=True):
    # define transforms
    trans = transforms.Compose([
    transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 归一化
    ])

    # load dataset
    dataset = datasets.CIFAR10(
        data_dir, train=False, download=False, transform=trans
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader
