import torch
import torch.nn as nn
import numpy as np
from function import split_train_data_to_tensor, shuffle_tensor
import sys

sys.path.append('/opt/anaconda3/lib/python3.7/site-packages/tensorflow/examples/tutorials/mnist')


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(66, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Sigmoid(),  # compress to 64 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 66),
            nn.Sigmoid(),  # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class AE(nn.Module):
    def __init__(self, size_input=66, size_hidden=64):
        super(AE, self).__init__()
        self.size_input = size_input
        self.size_hidden = size_hidden
        self.threshold = 0
        self.encoder = nn.Sequential(
            nn.Linear(self.size_input, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, self.size_hidden),
            nn.Sigmoid(),  # compress to 64 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, self.size_input),
            nn.Sigmoid(),  # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def fit(self, data, save_path, contamin=0.01, EPOCH=5, Batch_size=64,
            use_gpu=1, loss_func=nn.MSELoss()):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        train_num = int(len(data) * 0.8)
        train_data, valid_data = split_train_data_to_tensor(data, train_num, use_gpu)
        best_loss = 2
        if use_gpu:  # 1
            self = self.cuda()

        if len(train_data) < Batch_size:
            Batch_size = len(train_data)
        '''模型训练与验证'''
        for epoch in range(EPOCH):
            images = shuffle_tensor(train_data)
            '''训练一个周期，共iter_num次，每次使用一个小批量'''
            iter_num = int(len(images) / Batch_size)
            for i in range(iter_num):
                d = images[i * Batch_size:(i + 1) * Batch_size]
                encoded_x, decoded_x = self.forward(d)
                loss = loss_func(decoded_x, d)  # caculating this sum loss ,总损失
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

                '''每迭代50次，显示一次损失值,记录并保存最佳模型'''
                if i % 200 == 0:
                    print('epoch: ', epoch, 'iter: ', i, '| train loss: %.4f' % loss.data.cpu().numpy())
                    encoded_x, decoded_x = self.forward(valid_data)
                    loss_val = loss_func(decoded_x, valid_data)
                    print('previous best_val_loss: %.4f' % best_loss,
                          '| this_val_loss: %.4f' % loss_val.data.cpu().numpy())
                    if loss_val.data.cpu().numpy() < best_loss:
                        torch.save(self, save_path)  # 'ae.pkl'
                        best_loss = loss_val.data.cpu().numpy()
        encoded_x, decoded_x = self.forward(train_data)
        loss = [loss_func(decoded_x[i:i + 1], train_data[i:i + 1]).data.cpu().numpy() for i in range(train_num)]
        # threshold_local=int((1-contamin)*train_num)
        self.threshold = np.sort(np.array(loss))[int((1 - contamin) * train_num)]
        print('self.threshold', self.threshold)

    def predict(self, data, use_gpu=1, loss_func=nn.MSELoss()):
        print('self.threshold', self.threshold)
        if use_gpu:
            data = torch.Tensor(data).cuda()
            self = self.cuda()
        encoded_x, decoded_x = self.forward(data)
        loss = [loss_func(decoded_x[i:i + 1], data[i:i + 1]).data.cpu().numpy() for i in range(len(data))]
        results = [-1 if l > self.threshold else 1 for l in loss]
        return np.array(results), loss, encoded_x.data.cpu().numpy()


def get_ae_loss_feature(model1, model2, images, use_gpu=1):  # 对images,逐样本计算编码特征
    loss_func = nn.MSELoss()
    images_t = torch.Tensor(images)
    if use_gpu:
        images_t = images_t.cuda()
    _, decoded_data1 = model1(images_t)
    _, decoded_data2 = model2(images_t)
    loss1 = np.array([loss_func(decoded_data1[i:i + 1], images_t[i:i + 1]).data.cpu().numpy()
                      for i in range(len(images_t))])
    loss2 = np.array([loss_func(decoded_data2[i:i + 1], images_t[i:i + 1]).data.cpu().numpy()
                      for i in range(len(images_t))])
    loss1 = np.expand_dims(loss1, axis=1)
    loss2 = np.expand_dims(loss2, axis=1)
    return loss1, loss2, np.concatenate((loss1, loss2, images), axis=1)


def training_ae(model, optimizer, images_pos, images_neg, save_path, EPOCH=10, Batch_size=64,
                use_gpu=1, loss_func=nn.MSELoss()):
    best_loss = 2
    min_len = min(len(images_pos), len(images_neg))
    train_num = int(min_len * 0.8)
    test_num = min_len - train_num
    images_pos_train, images_pos_val = split_train_data_to_tensor(images_pos, train_num, use_gpu)
    images_neg_train, images_neg_val = split_train_data_to_tensor(images_neg, train_num, use_gpu)
    images_pos_val = images_pos_val[:test_num]
    images_neg_val = images_neg_val[:test_num]

    '''模型训练与验证'''
    for epoch in range(EPOCH):
        images_pos_t = shuffle_tensor(images_pos_train)
        images_neg_t = shuffle_tensor(images_neg_train)

        '''训练一个周期，共iter_num次，每次使用一个小批量'''
        iter_num = int(len(images_pos_t) / Batch_size)
        for i in range(iter_num):
            x_pos = images_pos_t[i * Batch_size:(i + 1) * Batch_size]
            x_neg = images_neg_t[i * Batch_size:(i + 1) * Batch_size]
            _, decoded_pos = model(x_pos)
            _, decoded_neg = model(x_neg)
            loss = loss_func(decoded_pos, x_pos) - loss_func(decoded_neg, x_neg)  # caculating this sum loss ,总损失
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            '''每迭代50次，显示一次损失值,记录并保存最佳模型'''
            if epoch % 5 == 0 and i % 200 == 0:
                print('epoch: ', epoch, 'iter: ', i, '| train loss: %.4f' % loss.data.cpu().numpy())
                _, decoded_pos = model(images_pos_val)
                _, decoded_neg = model(images_neg_val)
                loss_val = loss_func(decoded_pos, images_pos_val) - loss_func(decoded_neg, images_neg_val)
                print('previous best_val_loss: %.4f' % best_loss, '| this_val_loss: %.4f' % loss_val.data.cpu().numpy())
                if loss_val.data.cpu().numpy() < best_loss:
                    torch.save(model, save_path)  # 'ae.pkl'
                    best_loss = loss_val.data.cpu().numpy()
