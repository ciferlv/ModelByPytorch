import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from sklearn import datasets
import numpy as np
import math


class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=1, bias=True)

    def forward(self, input):
        self.output = F.sigmoid(self.linear(input))
        return self.output


def load_data():
    iris_ = datasets.load_iris()
    x = iris_.data[:, :2]
    y = (iris_.target != 0) * 1

    # 1 is twice than 0 in y, so I give weight 2 for data labeled '0' and weight 1 data labeled '1'.
    one_num = np.sum(y)
    zero_num = len(y) - one_num
    print("Zero Num: {}\tOne Num: {}".format(zero_num, one_num))
    weights = np.ones_like(y) + 1 - y

    shuffled_data = list(zip(x, y, weights))
    np.random.shuffle(shuffled_data)
    return shuffled_data


if __name__ == "__main__":

    shuffled_data = load_data()
    # 4/5 data for training, 1/5 data for testing.
    split_point = int(len(shuffled_data) * 4 / 5)

    epoch = 1000
    mini_batch = 10

    lg = LogisticRegression(2)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(lg.parameters())

    for idx_epoch in range(epoch):
        update_times = math.ceil(split_point / mini_batch)
        for idx in range(update_times):
            if idx == update_times - 1:
                train_data = np.array(shuffled_data[idx * mini_batch:split_point])
            else:
                train_data = np.array(shuffled_data[idx * mini_batch:(idx + 1) * mini_batch])

            x = torch.Tensor(list(train_data[:, 0]))
            y = torch.Tensor(list(train_data[:, 1]))
            criterion.weight = torch.Tensor(list(train_data[:, 2]))

            optimizer.zero_grad()
            loss = criterion(lg(x), y)
            loss.backward()
            optimizer.step()
            print("Epoch: {} MiniBatch: {} Loss: {}".format(idx_epoch, idx, loss.item()))

    # test
    test_data = np.array(shuffled_data[split_point:])
    x = torch.Tensor(list(test_data[:,0]))
    y = np.array(list(test_data[:,1]))
    predicated_y = (lg(x).squeeze(-1).detach().numpy() > 0.5) * 1
    precision = np.sum((y == predicated_y) * 1) / (len(shuffled_data) - split_point)
    print(y)
    print(predicated_y)
    print("Precision: {}".format(precision))
