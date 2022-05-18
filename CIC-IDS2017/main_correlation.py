import torch
import torch.nn as nn
import numpy as np
import random
import sys
from sklearn.preprocessing import OneHotEncoder

sys.path.append('/opt/anaconda3/lib/python3.7/site-packages/tensorflow/examples/tutorials/mnist')
use_gpu = torch.cuda.is_available()

'''定义封闭集检测模型'''


class Normal(nn.Module):
    def __init__(self, size_input=9, size_output=3):
        super(Normal, self).__init__()
        self.size_input = size_input
        self.size_output = size_output

        self.fc1 = nn.Linear(self.size_input, 8)
        self.afc1 = nn.Sigmoid()

        self.fc2 = nn.Linear(8, 6)
        self.afc2 = nn.Sigmoid()

        self.fc3 = nn.Linear(6, 3)

    def forward(self, x):
        x1 = self.afc1(self.fc1(x))
        x2 = self.afc2(self.fc2(x1))
        predict = self.fc3(x2)
        return x1, x2, predict

    def train(self, data, label, criterion, EPOCH=10, batch_size=64):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        iter_num = int(len(label) / batch_size)
        print(data.shape, label.shape)
        data = torch.Tensor(data)
        label = torch.Tensor(label[:, 0])
        for epoch in range(EPOCH):
            index = [i for i in range(len(label))]
            random.shuffle(index)
            data_x = data[index]
            label_l = label[index]
            for i in range(iter_num):
                x = data_x[i * batch_size:(i + 1) * batch_size]
                l = label_l[i * batch_size:(i + 1) * batch_size]
                x1, x2, predicts = self.forward(x.float())
                loss = criterion(predicts, l.long())

                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

            x1, x2, predicts = self.forward(data.float())
            predicts_y = predicts.argmax(dim=1)
            num_correct = torch.eq(predicts_y, label).sum().float().item()
            valid_acc = num_correct / len(label)
            print(valid_acc)
        torch.save(self, 'a1.pkl')  # 'ae1.pkl'

    def predcict(self, data):
        x1, x2, result = self.forward(data.float())
        return x1, x2, result, torch.cat((x1, x2, result), axis=1), result.argmax(dim=1)


t1 = np.zeros((50000, 1))
t2 = np.ones((50000, 1))
t3 = np.ones((50000, 1)) * 2
labels = np.concatenate((t1, t2, t3), axis=0)

ohe_encoder = OneHotEncoder(categories="auto")
t = ohe_encoder.fit_transform(labels).todense()

n1 = np.random.normal(0.2, 0.1, (50000, 6))  # np.random.random((50000, 9))
n2 = np.random.normal(0.2, 0.1, (50000, 6))  # np.random.random((50000, 9))
n3 = np.random.normal(0.2, 0.1, (50000, 6))  # np.random.random((50000, 9))
n = np.concatenate((n1, n2, n3), axis=0)

data1 = np.concatenate((t1, n1), axis=1)
data2 = np.concatenate((t2, n2), axis=1)
data3 = np.concatenate((t3, n3), axis=1)
data = np.concatenate((t, n), axis=1)

net = Normal()
for p in net.parameters():
    print(p)
net.train(data, labels, nn.CrossEntropyLoss())
for p in net.parameters():
    print(p)

n1 = np.random.normal(0.5, 0.1, (50000, 6))  # np.random.random((50000, 9))
n2 = np.random.normal(0.5, 0.1, (50000, 6))  # np.random.random((50000, 9))
n3 = np.random.normal(0.5, 0.1, (50000, 6))  # np.random.random((50000, 9))
n = np.concatenate((n1, n2, n3), axis=0)

data1 = np.concatenate((t1, n1), axis=1)
data2 = np.concatenate((t2, n2), axis=1)
data3 = np.concatenate((t3, n3), axis=1)
data = np.concatenate((t, n), axis=1)

net = torch.load('a1.pkl')
data_ = torch.Tensor(data)
x1, x2, result, d, l = net.predcict(data_)
d = result.data.numpy()
da = np.concatenate((data, d), axis=1)
'''
x1.backward(torch.ones(data_.size()).double())
grad_x1 = data_.grad.data.numpy()
x2.backward(torch.ones(data_.size()).double())
grad_x2 = data_.grad.data.numpy()
result.backward(torch.ones(data_.size()).double())
grad_result = data_.grad.data.numpy()
'''
index = [i for i in range(len(da))]
random.shuffle(index)
da = da[index]
c = np.corrcoef(da[:5000].T)
