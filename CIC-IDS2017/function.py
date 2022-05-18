import random
import torch
from sklearn.preprocessing import OneHotEncoder


def split_train_data_to_tensor(images, train_num, use_gpu=1):
    '''划分训练集与验证集'''
    images = shuffle_tensor(images)
    train_images = images[:train_num]
    val_images = images[train_num:]
    '''验证集to Tensor() and cuda()'''
    tensor_train = torch.Tensor(train_images)
    tensor_val = torch.Tensor(val_images)
    if use_gpu:
        tensor_train = tensor_train.cuda()
        tensor_val = tensor_val.cuda()
    return tensor_train, tensor_val


def split_train_data_to_tensor_label(images, train_num, use_gpu=1):
    '''划分训练集与验证集'''
    images = shuffle_tensor(images)
    labels = images[:, -1:]
    ohe_encoder = OneHotEncoder(categories="auto")
    labels = ohe_encoder.fit_transform(labels).todense()

    '''验证集to Tensor() and cuda()'''
    images = torch.Tensor(images)
    labels = torch.Tensor(labels)
    if use_gpu:
        images = images.cuda()
        labels = labels.cuda()
    return images[:train_num], images[train_num:], labels[:train_num], labels[train_num:]


def shuffle_tensor(tensors):
    index = [i for i in range(len(tensors))]
    random.shuffle(index)
    return tensors[index]


def shuffle_tensor_label(tensors, labels):
    index = [i for i in range(len(tensors))]
    random.shuffle(index)
    return tensors[index], labels[index]



