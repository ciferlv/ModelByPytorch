import torch.nn as nn
import torch
import numpy as np
import math

from TransE.Triple import Triple, Path

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


def load_data(file_path, idx2e, e2idx, idx2r, r2idx):
    triple_list = []
    head_set = set()
    relation_set = set()
    tail_set = set()
    triple_dict = {}
    with open(file_path, "r", encoding="UTF-8") as f:
        for line in f.readlines():
            head, relation, tail = line.strip().split()
            hidx = e2idx[head]
            ridx = r2idx[relation]
            tidx = e2idx[tail]
            head_set.add(hidx)
            tail_set.add(tidx)
            relation_set.add(ridx)
            if head not in triple_dict:
                triple_dict[hidx] = Triple(hidx)
            triple_dict[hidx].add_path(path=Path(ridx, tidx))
            triple_list.append([hidx, ridx, tidx])
    return triple_list, triple_dict, list(head_set), list(tail_set), list(relation_set)


def data2idx(file_list, folder):
    entity_set = set()
    relation_set = set()
    for file_name in file_list:
        with open(file_name, "r", encoding="UTF-8") as f:
            for line in f.readlines():
                h, r, t = line.strip().split()
                entity_set.add(h)
                entity_set.add(t)
                relation_set.add(r)
    with open(folder + "entity2Idx.txt", "w", encoding="UTF-8") as f:
        f.write("Total Num: {}\n".format(len(list(entity_set))))
        cnt = 0
        for entity in entity_set:
            f.write("{} {}\n".format(entity, cnt))
            cnt += 1
    with open(folder + "relation2Idx.txt", "w", encoding="UTF-8") as f:
        f.write("Total Num: {}\n".format(len(list(relation_set))))
        cnt = 0
        for relation in relation_set:
            f.write("{} {}\n".format(relation, cnt))
            cnt += 1


def load_e_r_idx(entity2idx_file, relation2idx_file):
    idx2e = {}
    e2idx = {}
    idx2r = {}
    r2idx = {}
    with open(entity2idx_file, "r", encoding="UTF-8") as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0: continue
            eName, eIdx = line.strip().split()
            idx2e[int(eIdx)] = eName
            e2idx[eName] = int(eIdx)
    with open(relation2idx_file, "r", encoding="UTF-8") as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0: continue
            rName, rIdx = line.strip().split()
            idx2r[rIdx] = rName
            r2idx[rName] = int(rIdx)
    return idx2e, e2idx, idx2r, r2idx


def isValid(head, relation, tail, triple_dict):
    for path in triple_dict[head].path_list:
        if path.rela == relation and path.tail == tail:
            return True
    return False


def train(triple_list, train_dict, train_head, train_tail, model):
    optimizer = torch.optim.Adam(te.parameters())
    entity_list = list(set(train_head.extend(train_tail)))
    epoch = 1000
    minibatch = 1000
    loop_times = math.ceil(len(triple_list) / minibatch)
    for epoch_i in range(epoch):
        for loop_i in range(loop_times):
            posi_head_list = []
            posi_tail_list = []
            nege_head_list = []
            nege_tail_list = []
            rela_list = []
            start = loop_i * minibatch
            if loop_i == loop_times - 1:
                end = len(triple_list)
            else:
                end = (loop_i + 1) * minibatch
            posi_train_triple = triple_list[start:end]
            for one_triple in posi_train_triple:
                posi_head_list.append(one_triple[0])
                posi_tail_list.append(one_triple[2])
                rela_list.append(one_triple[1])
                if np.random.randn() > 0.5:
                    nege_head_list.append(one_triple[0])
                    while True:
                        choosed_entity_idx = np.random.randint(0, len(entity_list))
                        tail = entity_list[choosed_entity_idx]
                        if not isValid(one_triple[0], one_triple[1], tail, train_dict):
                            nege_tail_list.append(tail)
                            break
                else:
                    nege_tail_list.append(one_triple[2])
                    while True:
                        choosed_entity_idx = np.random.randint(0, len(entity_list))
                        head = entity_list[choosed_entity_idx]
                        if not isValid(head, one_triple[1], one_triple[2], train_dict):
                            nege_tail_list.append(head)
                            break
            model.zero_grad()
            loss = te.trainModel(posi_head_list=posi_head_list, posi_tail_list=posi_tail_list, nege_head_list=nege_head_list,
                                 nege_tail_list=nege_tail_list, r_list=rela_list)
            loss.backward()
            optimizer.step()

        if epoch_i!=0 and (epoch_i % 20 ==0 or epoch_i == epoch -1):




if __name__ == "__main__":
    folder = "./data/FB15K-237/"
    train_file = folder + "train.txt"
    valid_file = folder + "valid.txt"
    test_file = folder + "test.txt"
    entity2Idx_file = folder + "entity2Idx.txt"
    relation2Idx_file = folder + "relation2Idx.txt"
    data2idx([train_file, valid_file, test_file], folder)
    idx2e, e2idx, idx2r, r2idx = load_e_r_idx(entity2Idx_file, relation2Idx_file)

    train_triple_list, train_dict, train_head, train_tail, train_relation = load_data(train_file, idx2e, e2idx, idx2r,
                                                                                      r2idx)
    valid_triple_list, valid_dict, valid_head, valid_tail, valid_relation = load_data(valid_file, idx2e, e2idx, idx2r,
                                                                                      r2idx)
    test_triple_list, test_dict, test_head, test_tail, test_relation = load_data(test_file, idx2e, e2idx, idx2r, r2idx)

    te = TransE(50, len(e2idx.keys()), len(r2idx), 1)
    train(train_triple_list, train_dict, train_head, train_tail, te)
    #
    # te.zero_grad()
    # p_head_list = [0, 1]
    # p_tail_list = [2, 3]
    # r_list = [0, 1]
    # n_head_list = [0, 1]
    # n_tail_list = [3, 2]
    # loss = te.trainModel(posi_head_list=p_head_list, posi_tail_list=p_tail_list, nege_head_list=n_head_list,
    #                      nege_tail_list=n_tail_list, r_list=r_list)
    # print(loss)
    # loss.backward()
    # optimizer.step()
