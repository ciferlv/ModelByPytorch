from Unit import Triple, Path


class Data2IDX:
    def __init__(self, folder):
        self.folder = folder
        self.train_file = folder + "train.txt"
        self.valid_file = folder + "valid.txt"
        self.test_file = folder + "test.txt"
        self.entity2idx_file = folder + "entity2Idx.txt"
        self.relation2idx_file = folder + "relation2Idx.txt"

        self.idx2e = {}
        self.e2idx = {}
        self.idx2r = {}
        self.r2idx = {}

    def data2idx(self, file_list):
        entity_set = set()
        relation_set = set()
        for file_name in file_list:
            with open(file_name, "r", encoding="UTF-8") as f:
                for line in f.readlines():
                    h, r, t = line.strip().split()
                    entity_set.add(h)
                    entity_set.add(t)
                    relation_set.add(r)
        with open(self.entity2idx_file, "w", encoding="UTF-8") as f:
            f.write("Total Num: {}\n".format(len(list(entity_set))))
            cnt = 0
            for entity in entity_set:
                f.write("{} {}\n".format(entity, cnt))
                cnt += 1
        with open(self.relation2idx_file, "w", encoding="UTF-8") as f:
            f.write("Total Num: {}\n".format(len(list(relation_set))))
            cnt = 0
            for relation in relation_set:
                f.write("{} {}\n".format(relation, cnt))
                cnt += 1

    def load_e_r_idx(self):
        with open(self.entity2idx_file, "r", encoding="UTF-8") as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0: continue
                eName, eIdx = line.strip().split()
                self.idx2e[int(eIdx)] = eName
                self.e2idx[eName] = int(eIdx)
        with open(self.relation2idx_file, "r", encoding="UTF-8") as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0: continue
                rName, rIdx = line.strip().split()
                self.idx2r[rIdx] = rName
                self.r2idx[rName] = int(rIdx)

    def triple2idx(self, triple_file, output_file):
        triple_idx_list = []
        with open(triple_file, "r", encoding="UTF-8") as f:
            for line in f.readlines():
                head, relation, tail = line.strip().split()
                hidx = self.e2idx[head]
                ridx = self.r2idx[relation]
                tidx = self.e2idx[tail]
                triple_idx_list.append([hidx, ridx, tidx])
        with open(output_file, "w", encoding="UTF-8") as f:
            for triple_idx in triple_idx_list:
                f.write("{}\t{}\t{}\n".format(triple_idx[0], triple_idx[1], triple_idx[2]))

    def er2idx(self):
        self.data2idx([self.train_file, self.valid_file, self.test_file])
        self.load_e_r_idx()


if __name__ == "__main__":
    folder = "./data/FB15k-237/"
    data2idx = Data2IDX(folder)
    # data2idx.er2idx()
    data2idx.load_e_r_idx()
    data2idx.triple2idx(folder + "test.txt", folder + "test_idx.txt")

