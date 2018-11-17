import torch.nn as nn
import torch
import numpy as np
import math
import random

from Unit import Triple, Path

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

    def get_e_embed(self, idx_list):
        look_up = torch.LongTensor(idx_list)
        return self.e_embed(look_up)

    def get_r_embed(self, idx_list):
        look_up = torch.LongTensor(idx_list)
        return self.r_embed(look_up)

    '''
    Given idx of head and idx of relation, predict the vector of tail
    Parameters:
    -----------
    h_idx: list, [[h_idx],[h_idx],...]
    r_idx: list, [[r_idx],[r_idx],...]
    
    Returns:
    -----------
    out: tensor, tensor of tail predicted
    '''

    def predict_tail_and_hits10(self, h_idx_list, r_idx_list, t_idx_list, e_list):
        print("Start testing")
        h_tensor = self.get_e_embed(h_idx_list)
        r_tensor = self.get_r_embed(r_idx_list)
        e_tensor = self.get_e_embed(e_list)

        predicated_t_tensor = h_tensor + r_tensor
        hits10 = 0
        for i in range(predicated_t_tensor.shape[0]):
            distance = torch.norm(predicated_t_tensor[i] - e_tensor, 2, -1, False)
            t_dis = distance[t_idx_list[i]]
            posi = torch.sum(distance<t_dis,-1)
            if posi < 9:
                hits10 += 1
        # t_idx_predicated = torch.argmin(distance, -1)
        # t_dis = torch.gather(distance, 1, torch.LongTensor(t_idx_list).unsqueeze(-1)).unsqueeze(-1)
        # hits10 = torch.sum(torch.sum(distance > t_dis, -1) < 9) / h_tensor.shape[0]

        print("Hits@10: {}".format(hits10 / len(h_idx_list)))

        # return t_idx_predicated, hits10
        return hits10


def load_data(triple_idx_file, e_idx_file, r_idx_file):
    e2idx = {}
    r2idx = {}
    with open(e_idx_file, "r", encoding="UTF-8") as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0: continue
            eName, eIdx = line.strip().split()
            # idx2e[int(eIdx)] = eName
            e2idx[eName] = int(eIdx)
    with open(r_idx_file, "r", encoding="UTF-8") as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0: continue
            rName, rIdx = line.strip().split()
            # idx2r[rIdx] = rName
            r2idx[rName] = int(rIdx)
    triple_list = []
    h_set = set()
    t_set = set()
    triple_dict = {}
    with open(triple_idx_file, "r", encoding="UTF-8") as f:
        for line in f.readlines():
            h_idx, r_idx, t_idx = list(map(int, line.strip().split()))
            h_set.add(h_idx)
            t_set.add(t_idx)
            if h_idx not in triple_dict:
                triple_dict[h_idx] = Triple(h_idx)
            triple_dict[h_idx].add_path(path=Path(r_idx, t_idx))
            triple_list.append([h_idx, r_idx, t_idx])
    return e2idx, r2idx, h_set, t_set, triple_list, triple_dict


def isValid(head, relation, tail, triple_dict):
    if head not in triple_dict:
        return True

    for path in triple_dict[head].path_list:
        if path.rela == relation and path.tail == tail:
            return True
    return False


def train(triple_list, train_dict, train_head_set, train_tail_set, model):
    optimizer = torch.optim.Adam(te.parameters())
    entity_list = list(train_head_set | train_tail_set)
    epoch = 50
    minibatch = 1000
    loop_times = math.ceil(len(triple_list) / minibatch)
    for epoch_i in range(epoch):
        loss_running = 0
        for loop_i in range(loop_times):
            # print("Epoch: {} Loop: {}".format(epoch_i, loop_i))
            posi_head_list = []
            posi_tail_list = []
            nege_head_list = []
            nege_tail_list = []
            rela_list = []
            start = loop_i * minibatch

            end = len(triple_list) if loop_i == loop_times - 1 else (loop_i + 1) * minibatch

            posi_train_triple = triple_list[start:end]
            for one_triple in posi_train_triple:
                posi_head_list.append(one_triple[0])
                posi_tail_list.append(one_triple[2])
                rela_list.append(one_triple[1])
                if np.random.randn() > 0.5:
                    nege_head_list.append(one_triple[0])
                    while True:
                        t_idx = random.sample(entity_list, 1)[0]
                        if not isValid(one_triple[0], one_triple[1], t_idx, train_dict):
                            nege_tail_list.append(t_idx)
                            break
                else:
                    nege_tail_list.append(one_triple[2])
                    while True:
                        h_idx = random.sample(entity_list, 1)[0]
                        if not isValid(h_idx, one_triple[1], one_triple[2], train_dict):
                            nege_head_list.append(h_idx)
                            break
            model.zero_grad()
            loss = te.trainModel(posi_head_list=posi_head_list, posi_tail_list=posi_tail_list,
                                 nege_head_list=nege_head_list,
                                 nege_tail_list=nege_tail_list, r_list=rela_list)
            loss_running += loss.item()
            loss.backward()
            optimizer.step()
        print("Epoch: {}, Loss: {}".format(epoch_i, loss_running))


if __name__ == "__main__":
    folder = "./data/FB15K-237/"
    train_idx_file = folder + "train_idx.txt"
    valid_idx_file = folder + "valid_idx.txt"
    test_idx_file = folder + "test_idx.txt"
    e2idx_file = folder + "entity2Idx.txt"
    r2idx_file = folder + "relation2Idx.txt"

    e2idx, r2idx, train_h_set, train_t_set, train_triple_list, train_triple_dict = load_data(train_idx_file, e2idx_file,
                                                                                             r2idx_file)
    _, _, valid_h_set, valid_t_set, valid_triple_list, valid_triple_dict = load_data(valid_idx_file, e2idx_file,
                                                                                     r2idx_file)
    _, _, test_h_set, test_t_set, test_triple_list, test_triple_dict = load_data(test_idx_file, e2idx_file, r2idx_file)

    te = TransE(50, len(e2idx.keys()), len(r2idx), 1)

    valid_h_idx_list = []
    valid_r_idx_list = []
    valid_t_idx_list = []
    for triple in valid_triple_list:
        valid_h_idx_list.append(triple[0])
        valid_r_idx_list.append(triple[1])
        valid_t_idx_list.append(triple[2])
    for round in range(20):
        print("Round {}.".format(round))
        train(train_triple_list, train_triple_dict, train_h_set, train_t_set, te)
        te.predict_tail_and_hits10(valid_h_idx_list, valid_r_idx_list, valid_t_idx_list, list(range(len(e2idx.keys()))))
