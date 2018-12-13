import numpy as np
import torch.nn as nn
import torch

x = np.random.randint(0, 10, (100, 2))
w = np.array([3, 4]).reshape((2, 1))
y = np.matmul(x, w)


class LR(nn.Module):
    def __init__(self, input_size):
        super(LR, self).__init__()
        self.linear = nn.Linear(input_size=input_size, out_features=1, bias=True)

