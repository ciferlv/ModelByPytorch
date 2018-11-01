import math


class Path:
    def __init__(self, rela, tail):
        self.rela = rela
        self.tail = tail

    def __str__(self):
        return str(self.rela) + "\t" + str(self.tail)


class Triple:
    def __init__(self, head):
        self.head = head
        self.path_list = []

    def add_path(self, path):
        self.path_list.append(path)


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


def train(triple_list, train_dict, train_head, train_tail, train_relation):
    epoch = 1000
    minibatch = 1000
    loop_times = math.ceil(len(triple_list) / minibatch)
    for epoch_i in range(epoch):
        for loop_i in range(loop_times):
            start = loop_i * minibatch
            if loop_i == loop_times - 1:
                end = len(triple_list)
            else:
                end = (loop_i + 1) * minibatch
            posi_train_triple = triple_list[start:end]


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
