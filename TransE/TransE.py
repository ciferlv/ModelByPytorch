import torch.nn as nn
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransE(nn.Module):
    def __init__(self, embed_dim, e_num, r_num, margin):
        super(TransE, self).__init__()
        low = -6 / np.sqrt(embed_dim)
        high = 6 / np.sqrt(embed_dim)
        e_embed = np.random.uniform(low=low, high=high, size=(e_num, embed_dim))
        self.e_embed = nn.Embedding(e_num, embed_dim, norm_type=2, _weight=torch.Tensor(e_embed))
        r_embed = np.random.uniform(low=low, high=high, size=(r_num, embed_dim))
        self.r_embed = nn.Embedding(r_num, embed_dim, norm_type=2, _weight=torch.Tensor(r_embed))
        self.margin = margin

    def trainModel(self, posi_head_list, posi_tail_list, nege_head_list, nege_tail_list, r_list):
        r = self.r_embed(torch.LongTensor(r_list))
        posi_head = self.e_embed(torch.LongTensor(posi_head_list))
        posi_tail = self.e_embed(torch.LongTensor(posi_tail_list))
        nege_head = self.e_embed(torch.LongTensor(nege_head_list))
        nege_tail = self.e_embed(torch.LongTensor(nege_tail_list))
        res = self.margin + torch.norm(posi_head + r - posi_tail, 2, -1, True) - torch.norm(nege_head + r - nege_tail,
                                                                                            2, -1, True)
        mask = (res > 0).type(torch.float32)
        loss = torch.sum(res * mask)
        return loss

    def get_embed(self, idx_list):
        look_up = torch.LongTensor(idx_list)
        return self.e_embed(look_up)


if __name__ == "__main__":
    te = TransE(10, 5, 3, 0.1)
    optimizer = torch.optim.Adam(te.parameters())
    te.zero_grad()
    p_head_list = [0, 1]
    p_tail_list = [2, 3]
    r_list = [0, 1]
    n_head_list = [0, 1]
    n_tail_list = [3, 2]
    loss = te.trainModel(posi_head_list=p_head_list, posi_tail_list=p_tail_list, nege_head_list=n_head_list,
                         nege_tail_list=n_tail_list, r_list=r_list)
    print(loss)
    loss.backward()
    optimizer.step()
