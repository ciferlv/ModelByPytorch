import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import optim
from sklearn.model_selection import train_test_split

from Utils import Util

util = Util()

# (name, preprocess, d_input_func) = ("Raw data", lambda data: data, lambda x: x)
(name, preprocess, d_input_func) = ("Data and variances", lambda data: decorate_with_diffs(data, 2.0), lambda x: x * 2)

class Discriminator(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Discriminator, self).__init__()
        self.h1 = nn.Linear(input_size, hidden_size, bias=True)
        self.h2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.h3 = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, input):
        z1 = F.elu(self.h1(input))
        z2 = F.elu(self.h2(z1))
        z3 = F.sigmoid(self.h3(z2))
        return z3


class Generator(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Generator, self).__init__()
        self.h1 = nn.Linear(input_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, hidden_size)
        self.h3 = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        z1 = F.elu(self.h1(input))
        z2 = F.sigmoid(self.h2(z1))
        z3 = self.h3(z2)
        return z3


def normal_sampler_function():
    return lambda m, n: np.random.normal(loc=4, scale=1.25, size=(m, n))


def uniform_sampler_function():
    return lambda m, n: torch.rand(m, n)

def decorate_with_diffs(data, exponent):
    mean = torch.mean(data.data, 1, keepdim=True)
    mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
    diffs = torch.pow(data - torch.Tensor(mean_broadcast), exponent)
    return torch.cat([data, diffs], 1)

def train_GAN():
    d_input_size = 100
    d_hidden_size = 50
    d_output_size = 1
    g_input_size = 1
    g_hidden_size = 50
    g_output_size = 1
    normal_sampler = normal_sampler_function()
    uniform_sampler = uniform_sampler_function()
    D = Discriminator(input_size=d_input_func(d_input_size), output_size=d_output_size, hidden_size=d_hidden_size)
    G = Generator(input_size=g_input_size, output_size=g_output_size, hidden_size=g_hidden_size)
    d_optm = torch.optim.Adam(D.parameters())
    g_optm = torch.optim.Adam(G.parameters())
    criterion = torch.nn.BCELoss()
    D_steps = 1
    G_steps = 1
    epoch = 30000
    mean_list = []
    std_list = []
    for epoch_i in range(epoch):
        print("Epoch: {}".format(epoch_i))
        for d_i in range(D_steps):
            # train on real
            D.zero_grad()
            train_x_real = torch.Tensor(normal_sampler(1, d_input_size))
            train_y_real = torch.ones(1)
            output = D(preprocess(train_x_real))
            real_loss = criterion(output, train_y_real)
            real_loss.backward()

            # train on fake
            fake_train_x = uniform_sampler(d_input_size, 1)
            generated_uniform = G(fake_train_x).detach()
            discrimiated_result = D(preprocess(generated_uniform.t()))
            fake_loss = criterion(discrimiated_result, torch.zeros(1))
            fake_loss.backward()
            d_optm.step()
            mean_list.append(np.mean(generated_uniform.data.storage().tolist()))
            std_list.append(np.std(generated_uniform.data.storage().tolist()))
            print("Mean: {} Devi: {}".format(np.mean(generated_uniform.data.storage().tolist()),
                                             np.std(generated_uniform.data.storage().tolist())))

        for g_i in range(G_steps):
            G.zero_grad()
            fake_train_x = uniform_sampler(d_input_size, 1)
            generated_uniform = G(fake_train_x)
            discrimiated_result = D(preprocess(generated_uniform.t()))
            fake_loss = criterion(discrimiated_result, torch.ones(1))
            fake_loss.backward()
            g_optm.step()
    util.plot_scatter_diagram("Mean by epoch.", "Epoch", "Mean", np.arange(len(mean_list)), mean_list)
    util.plot_scatter_diagram("Std by epoch.", "Epoch", "Std", np.arange(len(std_list)), std_list)


def test_Discriminator():
    posi_x = np.random.normal(loc=4, scale=1.25, size=(1000, 5))
    posi_y = np.ones(posi_x.shape[0])
    nege_x = np.random.uniform(0, 1, 1000, 5)
    nege_y = np.zeros(nege_x.shape[0])

    x = np.append(posi_x, nege_x, axis=0)
    y = np.append(posi_y, nege_y)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, shuffle=True)

    d = Discriminator(5, 1)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(d.parameters())
    mini_batch = 50
    epoch = 100
    for epoch_i in range(epoch):
        train_x_splited, train_y_splited = util.split_by_mini_batch(mini_batch, train_x, train_y)
        for idx in range(len(train_x_splited)):
            optimizer.zero_grad()
            output = d(torch.Tensor(train_x_splited[idx]))
            loss = criterion(output, torch.Tensor(train_y_splited[idx]))
            loss.backward()
            optimizer.step()
            print("Epoch: {} Mini Batch: {} Loss: {}.".format(epoch_i, idx, loss.item()))

    test_output = d(torch.Tensor(test_x))
    predicated_y = (test_output.squeeze(-1).detach().numpy() > 0.5) * 1
    precision = np.sum((test_y == predicated_y) * 1) / len(test_y)
    print("Precision: {}".format(precision))


if __name__ == "__main__":
    # test_Discriminator()
    train_GAN()
