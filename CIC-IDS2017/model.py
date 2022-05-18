import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributions as D
from sklearn.preprocessing import OneHotEncoder
import random
import sys
from function import split_train_data_to_tensor, shuffle_tensor, split_train_data_to_tensor_label, shuffle_tensor_label

sys.path.append('/opt/anaconda3/lib/python3.7/site-packages/tensorflow/examples/tutorials/mnist')
use_gpu = torch.cuda.is_available()

'''定义封闭集检测模型'''


class Normal(nn.Module):
    def __init__(self, size_input=71, size_output=2):
        super(Normal, self).__init__()
        self.size_input = size_input
        self.size_output = size_output
        self.net = nn.Sequential(
            nn.Linear(self.size_input, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, self.size_output),
        )

    def forward(self, x):
        predict = self.net(x)
        return predict


class Doc(nn.Module):
    def __init__(self, size_input=71, size_output=2):
        super(Doc, self).__init__()
        self.size_input = size_input
        self.size_output = size_output
        self.net = nn.Sequential(
            nn.Linear(self.size_input, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, self.size_output),
            nn.Sigmoid(),
        )

    def forward(self, x):
        predict = self.net(x)
        return predict


class CvaeEvt(nn.Module):
    def __init__(self, size_input=71, size_output=2, size_hidden=3):
        super(CvaeEvt, self).__init__()
        self.size_input = size_input
        self.size_output = size_output
        self.size_hidden = size_hidden
        self.staus = 1  # 1 train_step1,2 train step 2, 3 test
        self.fc11 = nn.Linear(18, self.size_hidden)
        self.fc12 = nn.Linear(18, self.size_hidden)
        self.fc21 = nn.Linear(20, self.size_hidden)
        self.fc22 = nn.Linear(20, self.size_hidden)

        self.PN = nn.Sequential(
            nn.Linear(self.size_input, 58),
            nn.Softplus(),
            nn.Linear(58, 38),
            nn.Softplus(),
            nn.Linear(38, 18),
            nn.Softplus(),
            nn.Linear(18, 2 * 3),
        )

        self.RN = nn.Sequential(
            nn.Linear(self.size_input + self.size_output, 60),
            nn.Softplus(),
            nn.Linear(60, 40),
            nn.Softplus(),
            nn.Linear(40, 20),
            nn.Softplus(),
            nn.Linear(20, 2 * 3),
        )
        self.cls = nn.Sequential(
            nn.Linear(self.size_input + 3, 61),
            nn.Softplus(),
            nn.Linear(61, 41),
            nn.Softplus(),
            nn.Linear(41, 21),
            nn.Softplus(),
            nn.Linear(21, self.size_output),
            nn.Sigmoid()
        )
        self.GN = nn.Sequential(
            nn.Linear(3 + self.size_output, 22),
            nn.Softplus(),
            nn.Linear(22, 32),
            nn.Softplus(),
            nn.Linear(32, 42),
            nn.Softplus(),
            nn.Linear(42, 52),
            nn.Softplus(),
            nn.Linear(52, 68),
            nn.Softplus(),
            nn.Linear(68, self.size_input)
        )

    def set_stau(self, stau):
        self.staus = stau
        if self.staus == 2:
            for p in self.RN.parameters():
                p.requires_grad = False

    def decoder_PN(self, x):
        mu, logsigma = self.PN(x).chunk(2, dim=-1)
        return D.Normal(mu, logsigma.exp())

    def decoder_RN(self, x):
        mu, logsigma = self.RN(x).chunk(2, dim=-1)
        return D.Normal(mu, logsigma.exp())

    def forward(self, x, y=None):
        if self.staus == 1:
            xy = torch.cat((x, y), 1)
            z_xy = self.decoder_RN(xy)
            z_x = self.decoder_PN(x)
            z = z_xy.rsample()
            xz = torch.cat((x, z), 1)
            predict = self.cls(xz)
            return z_xy, z_x, predict
        elif self.staus == 2:
            xy = torch.cat((x, y), 1)
            z_xy = self.decoder_RN(xy)
            z = z_xy.rsample()
            yz = torch.cat([y, z], 1)
            x_yz = self.GN(yz)
            return x_yz
        else:
            z_x = self.decoder_PN(x)
            z = z_x.rsample()
            xz = torch.cat([x, z], dim=1)
            predict = self.cls(xz)
            yz = torch.cat([predict, z], dim=1)
            x_recon = self.GN(yz)
            return predict, x_recon


class crosr_net(nn.Module):
    def __init__(self, size_input=71, size_output=2, size_hidden=3):
        super(crosr_net, self).__init__()
        self.size_input = size_input
        self.size_output = size_output
        self.size_hidden = size_hidden
        self.staus = 1  # 1 train_step1,2 train step 2, 3 test

        self.fc1 = nn.Linear(self.size_input, 32)
        self.afc1 = nn.ReLU()

        self.fc2 = nn.Linear(32, 16)
        self.afc2 = nn.ReLU()

        self.fc3 = nn.Linear(16, self.size_output)

        self.g2 = nn.Linear(16 + 8, 32)
        self.ag2 = nn.ReLU()

        self.g1 = nn.Linear(32 + 8, self.size_input)
        self.ag1 = nn.Sigmoid()

        self.fh1 = nn.Linear(self.size_input, 8)
        self.afh1 = nn.Sigmoid()

        self.fh2 = nn.Linear(32, 8)
        self.afh2 = nn.Sigmoid()

        self.fh3 = nn.Linear(16, 8)
        self.afh3 = nn.Sigmoid()

        self.gh1 = nn.Linear(8, 16)
        self.agh1 = nn.ReLU()

    def forward(self, x):
        x2 = self.afc1(self.fc1(x))
        x3 = self.afc2(self.fc2(x2))
        y_ = self.fc3(x3)
        z1 = self.afh1(self.fh1(x))
        z2 = self.afh2(self.fh2(x2))
        z3 = self.afh3(self.fh3(x3))
        x3_ = self.agh1(self.gh1(z3))
        x2_ = self.ag2(self.g2(torch.cat([x3_, z2], dim=1)))
        x_ = self.ag1(self.g1(torch.cat([x2_, z1], dim=1)))
        return y_, torch.cat([y_, z1, z2, z3], dim=1), x_


'''计算样本分类输出特征'''


def get_cls_feature(ae_feature, cls_path):  # 对所有数据,逐样本计算重建损失,用于测试中损失分布的计算.计算
    model = torch.load(cls_path)
    ae_feature = torch.Tensor(ae_feature)
    if use_gpu:  # 1
        model = model.cuda()
        ae_feature = ae_feature.cuda()
    pred = model(ae_feature)
    # soft_pred=F.softmax(pred, dim=1)
    return pred.data.cpu().numpy()


def get_cls_act_pre_soft(data_training, cls_path):
    # print('--get feature from  :',cls_path)
    model = torch.load(cls_path)
    data_training = torch.Tensor(data_training)
    if use_gpu:  # 1
        model = model.cuda()
        data_training = data_training.cuda()
    activations = model(data_training)
    softmax_pred = F.softmax(activations, dim=1).data.cpu().numpy()
    activations = activations.data.cpu().numpy()
    predicts = np.argmax(activations, 1)
    return activations, predicts, softmax_pred


def get_output_cvae_evt(cls_path, data, label=[], stau=3):
    loss_func = nn.MSELoss()
    # print(cls_path)
    model = torch.load(cls_path)

    data = torch.Tensor(data).float()
    if len(label) != 0:
        label = torch.Tensor(label).long()
        label = torch.unsqueeze(label, 1)
        label = torch.zeros(label.size()[0], model.size_output).scatter_(1, label, 1)
    if use_gpu:  # 1
        model = model.cuda()
        data = data.cuda()
        if len(label) != 0:
            label = label.cuda()

    model.set_stau(stau)
    if model.staus == 1:
        z_xy, z_x, pred = model(data, label)
        return pred.data.cpu().numpy()
    elif model.staus == 2:
        x_recon = model(data, label)
        loss = [loss_func(x_recon[i:i + 1], data[i:i + 1]).data.cpu().numpy() for i in
                range(len(data))]
        return loss
    else:
        predicts, x_recon = model(data)
        loss = [loss_func(x_recon[i:i + 1], data[i:i + 1]).data.cpu().numpy() for i in
                range(len(data))]
        return predicts.data.cpu().numpy(), loss


'''训练分类模型'''


def get_centers(f_t, label, model):
    pred = model(f_t)
    pred_y2 = torch.argmax(pred, dim=1)
    # print(pred.shape,pred_y2.shape)
    mean = []
    for i in torch.unique(label):
        data = pred[pred_y2 == i]
        if len(data) != 0:
            mean.append(torch.unsqueeze(torch.mean(pred[pred_y2 == i], axis=0).detach(), 0))
        else:
            mean.append(torch.zeros_like(pred[0:1]))
    mean = torch.cat(mean, dim=0)
    return mean


def get_centers_ocn(f_t, label, model, centers):
    pred = model(f_t)
    pred_y2 = torch.argmax(pred, dim=1)
    mean = []
    for i in range(len(np.unique(label))):
        mean.append(np.mean(np.array(pred[pred_y2 == i]), axis=0))

    if not torch.sum(centers):
        centers = mean
    else:
        for i in range(len(mean)):
            if torch.sum(pred_y2 == i):
                centers[i] = 0.5 * centers[i] + 0.5 * np.mean(np.array(pred[pred_y2 == i]), axis=0)
    return centers


def criterion_ii2(pred, x_l, criterion):
    criterion1 = torch.nn.MSELoss()
    loss1 = criterion(pred, x_l)

    t = torch.unique(x_l, sorted=False)
    centers = [torch.mean(pred[x_l == x], axis=0) for x in t]

    if use_gpu:  # 1
        all_center = torch.empty(len(x_l), len(pred[0])).cuda()
    else:
        all_center = torch.empty(len(x_l), len(pred[0]))
    for i, x in enumerate(t):
        all_center[x_l == x] = centers[i]

    loss3 = 0
    for i in centers:
        for j in centers:
            loss3 = loss3 + criterion1(i, j)
    loss2 = criterion1(pred, all_center.float()) - loss3
    loss = loss1 + loss2
    return loss


def criterion_ii(pred, x_l, centers, criterion):
    loss1 = criterion(pred, x_l.long())
    all_center = [centers[i:i + 1] for i in x_l]
    all_center = torch.cat(all_center, dim=0)
    criterion1 = torch.nn.MSELoss()
    if use_gpu:
        loss2 = criterion1(pred, all_center)
    else:
        loss2 = criterion1(pred, torch.Tensor(all_center))

    loss3 = 0
    for i in range(len(centers)):
        for j in range(len(centers)):
            loss3 = loss3 + criterion1(centers[i], centers[j])
    loss = loss1 + loss2 + loss3
    return loss


def criterion_centerids(pred, x_l, centers, criterion):
    loss1 = criterion(pred, x_l.long())
    all_center = [centers[i:i + 1] for i in x_l]
    criterion1 = torch.nn.MSELoss()
    if use_gpu:
        loss2 = criterion1(pred, all_center)
    else:
        loss2 = criterion1(pred, torch.Tensor(all_center))
    loss = loss1 + loss2
    return loss


import math


def criterion_cac(pred, x_l, centers, criterion):
    all_center = [centers[i:i + 1] for i in x_l]
    loss1 = criterion(pred, all_center)
    distance = [criterion(pred[i].repeat(len(pred)), all_center) for i in range(len(pred))]
    e_distance = []
    for i in range(len(pred)):
        e_distance.append(sum([math.exp(distance[x_l[i]], distance[j]) for j in range(len(pred))]) - 1)
    loss2 = math.log(1 + e_distance)
    loss = loss1 + loss2
    return loss


def criterion_ocn(pred, x_l, centers, criterion):
    criterion1 = torch.nn.MSELoss()
    loss1 = criterion(pred, x_l.long())
    all_center = [centers[i] for i in x_l.cpu.numpy()]
    if use_gpu:  # 1
        all_center = all_center.cuda()
    loss3 = 0
    for i in range(len(centers)):
        loss3 = loss3 + torch.sum([criterion1(centers[i], centers[j]) for j in centers])
    loss2 = criterion1(pred, all_center.float()) - loss3
    loss = loss1 + 0.05 * loss2
    return loss


def get_loss(model, data, is_batch, loss_func):  # 训练中，对每个批量计算损失

    _, decoded_data = model(data)
    if is_batch:
        loss = loss_func(decoded_data, data)  # mean square error ，正例重建损失    # caculating this sum loss ,总损失
    else:
        loss = [loss_func(decoded_data[i:i + 1], data[i:i + 1]).cpu().numpy() for i in range(len(data))]
    return loss


def loss_criterion(method, predicts, labels, criterion, centers):
    if method == 'centerids':
        loss = criterion_centerids(predicts, labels, centers, criterion)
    elif method == 'center_ii':
        # print(type(labels.cpu()))
        loss = criterion_ii(predicts, labels, centers, criterion)
    elif method == 'center_ocn':
        loss = criterion_ocn(predicts, labels, centers, criterion)
    elif method == 'cac':
        loss = criterion_cac(predicts, labels, centers, criterion)
    else:
        loss = criterion(predicts, labels)

    return loss


def add_gaussian_noise(tensor):
    # 产生高斯 noise
    noise = torch.normal(0, 1, tensor.shape)
    if use_gpu:
        noise = noise.cuda()
    # 将噪声和图片叠加
    gaussian_out = tensor + noise
    # 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = torch.clip(gaussian_out, 0, 1)
    return gaussian_out


def training_class(model, optimizer, f_training, label_training, EPOCH=10, batch_half=64, save_path='class.pkl',
                   method='not_centeids'):
    f_training = np.concatenate((f_training, np.expand_dims(label_training, axis=1)), axis=1)
    train_f, test_f_te = split_train_data_to_tensor(f_training, int(len(f_training) * 0.8), use_gpu)

    start_loss = 2000
    test_label_te = test_f_te[:, -1]
    test_f_te = test_f_te[:, :model.size_input]
    if method == 'center_ocn':
        test_f_te_noise = add_gaussian_noise(test_f_te)
        criterion_noise = nn.MSELoss()
    ohe_encoder = OneHotEncoder(categories="auto")
    lb = np.arange(len(np.unique(label_training))).reshape(len(np.unique(label_training)),1)
    centers = ohe_encoder.fit_transform(lb).todense()
    '''模型训练与验证'''
    for epoch in range(EPOCH):
        '''每个迭代shuffle训练集'''
        f_t = shuffle_tensor(train_f)
        label_t = f_t[:, -1]
        f_t = f_t[:, :model.size_input]
        if method in ['centerids', 'center_ii']:
            centers = get_centers(f_t, label_t, model)
        '''训练一个周期，共iter_num次，每次使用一个小批量'''
        iter_num = int(len(label_t) / batch_half)
        for i in range(iter_num):
            x_f = f_t[i * batch_half:(i + 1) * batch_half]
            x_l = label_t[i * batch_half:(i + 1) * batch_half]
            pred = model(x_f)
            if method == 'center_ocn':
                centers = get_centers_ocn(f_t, label_t, model, centers).detach()
                x_f_noise = add_gaussian_noise(x_f)
                pred_noise = model(x_f_noise)
                # print(pred.shape,pred_noise.shape)
                loss = loss_criterion(method, pred, x_l.long(), nn.CrossEntropyLoss(), centers) + criterion_noise(pred,
                                                                                                                  pred_noise)
            else:
                loss = loss_criterion(method, pred, x_l.long(), nn.CrossEntropyLoss(), centers)
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            '''每迭代50次，显示一次损失值,记录并保存最佳模型'''
            if i % 200 == 0:
                print('epoch: ', epoch, 'iter: ', i, '| train loss: %.4f' % loss.data.cpu().numpy())
                predicts = model(test_f_te)
                predicts_y = predicts.argmax(dim=1)
                num_correct = torch.eq(predicts_y, test_label_te).sum().float().item()
                valid_acc = num_correct / len(test_label_te)
                if method == 'center_ocn':
                    pred_noise = model(test_f_te_noise)
                    noise_loss = criterion_noise(predicts, pred_noise)
                    loss_t = loss_criterion(method, predicts, test_label_te.long(), nn.CrossEntropyLoss(),
                                            centers) + noise_loss
                else:
                    loss_t = loss_criterion(method, predicts, test_label_te.long(), nn.CrossEntropyLoss(), centers)
                print('best_val_loss', start_loss, 'this_val_loss', loss_t.data.cpu().numpy(), 'valid acc:', valid_acc)
                if loss_t.data.cpu().numpy() < start_loss:
                    torch.save(model, save_path)  # 'ae1.pkl'
                    start_loss = loss_t.data.cpu().numpy()


def criterion_cvae(z_xy, z_x, y_xz, y):
    KLD = D.kl.kl_divergence(z_xy, z_x)
    # KLD=torch.sum(KLD)
    loss = nn.BCELoss(reduction='sum').cuda()
    BCE = loss(y_xz, y)
    return (torch.sum(KLD) + BCE) / y.size(0)


def training_cvae_cls(model, optimizer, f_training, EPOCH=10, batch_half=64, save_path='class.pkl'):
    train_f, test_f_te, train_label_te, test_label_te = split_train_data_to_tensor_label(f_training,
                                                                                         int(len(f_training) * 0.8),
                                                                                         use_gpu)
    label_training = f_training[:, -1]
    f_training = f_training[:, :-1]
    index = [i for i in range(len(label_training))]
    random.shuffle(index)
    f_training = f_training[index]
    label_training = label_training[index]
    train_num = int(len(f_training) * 0.8)
    data = f_training[:train_num]
    label = label_training[:train_num]
    test_data = f_training[train_num:]
    test_label = label_training[train_num:]
    test_data = torch.Tensor(test_data).float()
    test_label = torch.Tensor(test_label).long()
    test_l = torch.unsqueeze(test_label, 1)
    test_l = torch.zeros(test_l.size()[0], model.size_output).scatter_(1, test_l, 1)
    if use_gpu:
        test_data = test_data.cuda()
        test_label = test_label.cuda()
        test_l = test_l.cuda()
    model.train()
    '''每个迭代shuffle训练集'''

    def tran_step(test_data, test_label, data, label, stau, start_loss, start_loss2, criterion):
        start_acc = 0
        f_t, label_t = shuffle_tensor_label(data, label)
        model.set_stau(stau)
        '''训练一个周期，共iter_num次，每次使用一个小批量'''
        iter_num = int(len(label_t) / batch_half)
        for i in range(iter_num):
            x_f = f_t[i * batch_half:(i + 1) * batch_half]
            x_l = label_t[i * batch_half:(i + 1) * batch_half]
            data = torch.Tensor(x_f).float()
            label = torch.Tensor(x_l).long()
            label = torch.unsqueeze(label, 1)
            label = torch.zeros(label.size()[0], model.size_output).scatter_(1, label, 1)
            if use_gpu:
                data = data.cuda()
                label = label.cuda()
            if model.staus == 1:
                z_xy, z_x, pred = model(data, label)
                loss = criterion(z_xy, z_x, pred, label)
            if model.staus == 2:
                x_recon = model(data, label)
                loss = criterion(x_recon, data)
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            '''每迭代50次，显示一次损失值,记录并保存最佳模型'''
            if i % 200 == 0:
                print('epoch: ', epoch, 'iter: ', i, '| train loss: %.4f' % loss.data.cpu().numpy())
                if model.staus == 1:
                    model.staus = 3
                    predicts_y, x_recon = model(test_data)
                    predicts_y = predicts_y.argmax(dim=1)
                    print(predicts_y.shape, test_label.shape)
                    num_correct = torch.eq(predicts_y, test_label).sum().float().item()
                    valid_acc = num_correct / len(test_label_te)
                    print('valid_acc', valid_acc)
                    print('best_valid_acc', start_acc, 'this_loss', loss.data.cpu().numpy())
                    if start_acc < valid_acc:
                        torch.save(model, save_path)  # 'ae1.pkl'
                        start_acc = valid_acc
                    model.staus = 1
                if model.staus == 2:
                    x_recon = model(test_data, test_label)
                    loss_t = criterion(x_recon, test_data)
                    print('best_val_loss', start_loss2, 'this_val_loss', loss_t.data.cpu().numpy())
                    if loss_t.data.cpu().numpy() < start_loss2:
                        torch.save(model, save_path)  # 'ae1.pkl'
                        start_loss2 = loss_t.data.cpu().numpy()

    '''模型训练与验证'''
    for epoch in range(EPOCH):
        tran_step(test_data, test_label, data, label, stau=1, start_loss=2000, start_loss2=2000,
                  criterion=criterion_cvae)
    for epoch in range(EPOCH):
        tran_step(test_data, test_l, data, label, stau=2, start_loss=2000, start_loss2=2000,
                  criterion=torch.nn.MSELoss())


def training_doc(model, optimizer, f_training, label_training, EPOCH=10, batch_half=64, save_path='class.pkl'):
    criterion = nn.BCELoss()

    index = [i for i in range(len(label_training))]
    random.shuffle(index)

    f_training = f_training[index]
    label_training = label_training[index]
    label_training = np.reshape(label_training, (len(label_training), -1))
    ohe_encoder = OneHotEncoder(categories="auto")
    label_training = ohe_encoder.fit_transform(label_training).todense()

    """
    #观察是否可以注释掉
    label_training = np.reshape(label_training, (len(label_training), -1))
    ohe_encoder = OneHotEncoder(categories="auto")
    label_training = ohe_encoder.fit_transform(label_training).todense()
    """
    # print(label_training[:10],label_training[-10:])
    '''划分训练集与验证集'''
    train_f = f_training[:int(len(label_training) * 0.8)]
    test_f = f_training[int(len(label_training) * 0.8):]
    train_label = label_training[:int(len(label_training) * 0.8)]
    test_label = label_training[int(len(label_training) * 0.8):]
    start_loss = 2000

    '''验证集'''
    test_f_te = torch.Tensor(test_f)
    test_label_te = torch.Tensor(test_label)
    train_f = torch.Tensor(train_f)
    train_label = torch.Tensor(train_label)
    if use_gpu:
        test_f_te = test_f_te.cuda()
        test_label_te = test_label_te.cuda()
        train_f = train_f.cuda()
        train_label = train_label.cuda()

    '''模型训练与验证'''
    for epoch in range(EPOCH):
        '''每个迭代shuffle训练集'''
        index = [i for i in range(len(train_label))]
        random.shuffle(index)
        f_t = train_f[index]
        label_t = train_label[index]

        '''训练一个周期，共iter_num次，每次使用一个小批量'''
        iter_num = int(len(label_t) / batch_half)
        for i in range(iter_num):
            x_f = f_t[i * batch_half:(i + 1) * batch_half]
            x_l = label_t[i * batch_half:(i + 1) * batch_half]
            pred = model(x_f)

            loss = criterion(pred, x_l.float())  # calculating this loss,总损失，正例重建损失，负例重建损失
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            '''每迭代50次，显示一次损失值,记录并保存最佳模型'''
            if i % 200 == 0:
                print('epoch: ', epoch, 'iter: ', i, '| train loss: %.4f' % loss.data.cpu().numpy())
                predicts = model(test_f_te)
                predicts_y = predicts.argmax(dim=1)
                num_correct = torch.eq(predicts_y, test_label_te.argmax(dim=1)).sum().float().item()
                valid_acc = num_correct / len(test_label_te)
                loss_t = criterion(predicts, test_label_te.float())
                print('best_val_loss', start_loss, 'this_val_loss', loss_t.data.cpu().numpy(), 'valid acc:', valid_acc)
                if loss_t.data.cpu().numpy() < start_loss:
                    torch.save(model, save_path)  # 'ae1.pkl'
                    start_loss = loss_t.data.cpu().numpy()
                # print(x_f[i][:10],pred[i])


def training_crosr(model, optimizer, f_training, label_training, EPOCH=10, batch_half=64, save_path='class.pkl'):
    f_training = np.concatenate((f_training, np.expand_dims(label_training, axis=1)), axis=1)
    train_f, test_f_te = split_train_data_to_tensor(f_training, int(len(f_training) * 0.8), use_gpu)
    criterion1 = nn.MSELoss()
    criterion2 = nn.CrossEntropyLoss()
    start_loss = 2000
    test_label_te = test_f_te[:, -1]
    test_f_te = test_f_te[:, :model.size_input]

    '''模型训练与验证'''
    for epoch in range(EPOCH):
        '''每个迭代shuffle训练集'''
        f_t = shuffle_tensor(train_f)
        label_t = f_t[:, -1]
        f_t = f_t[:, :model.size_input]
        '''训练一个周期，共iter_num次，每次使用一个小批量'''
        iter_num = int(len(label_t) / batch_half)
        for i in range(iter_num):
            x_f = f_t[i * batch_half:(i + 1) * batch_half]
            x_l = label_t[i * batch_half:(i + 1) * batch_half]
            y_, z_, x_ = model(x_f)
            loss = criterion2(y_, x_l.long()) + criterion1(x_, x_f)
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            '''每迭代50次，显示一次损失值,记录并保存最佳模型'''
            if i % 200 == 0:
                print('epoch: ', epoch, 'iter: ', i, '| train loss: %.4f' % loss.data.cpu().numpy())
                y_, z_, x_ = model(test_f_te)
                predicts_y = y_.argmax(dim=1)
                num_correct = torch.eq(predicts_y, test_label_te).sum().float().item()
                valid_acc = num_correct / len(test_label_te)
                loss_t = criterion2(y_, test_label_te.long()) + criterion1(x_, test_f_te)
                print('best_val_loss', start_loss, 'this_val_loss', loss_t.data.cpu().numpy(), 'valid acc:', valid_acc)
                if loss_t.data.cpu().numpy() < start_loss:
                    torch.save(model, save_path)  # 'ae1.pkl'
                    start_loss = loss_t.data.cpu().numpy()
