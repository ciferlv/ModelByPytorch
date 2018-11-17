import numpy as np
import torch

a = torch.Tensor([[5, 2, 3], [9, 4, 8], [1, 9, 1]])
b = torch.Tensor([[1, 2, 3], [7, 4, 8]])
idx = torch.LongTensor([[0], [2]])
print(a.shape)
print(b.shape)
distance = torch.norm(a - b.unsqueeze(1), 2, -1, False)
t_idx_predicated = torch.argmin(distance, -1)
print(distance)
print(t_idx_predicated)
t_dis = torch.gather(distance, 1, idx)
print(torch.sum(torch.sum(distance > t_dis,-1)<9) / b.shape[0])
print(distance > t_dis)

# print(res.squeeze(-1))
