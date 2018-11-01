from TransE.Triple import Triple, Path


class Conductor:
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

    def load_data(self,file_path):
        triple_list = []
        head_set = set()
        relation_set = set()
        tail_set = set()
        triple_dict = {}
        with open(file_path, "r", encoding="UTF-8") as f:
            for line in f.readlines():
                head, relation, tail = line.strip().split()
                hidx = self.e2idx[head]
                ridx = self.r2idx[relation]
                tidx = self.e2idx[tail]
                head_set.add(hidx)
                tail_set.add(tidx)
                relation_set.add(ridx)
                if head not in triple_dict:
                    triple_dict[hidx] = Triple(hidx)
                triple_dict[hidx].add_path(path=Path(ridx, tidx))
                triple_list.append([hidx, ridx, tidx])
        return triple_list, triple_dict, list(head_set), list(tail_set), list(relation_set)

    def process(self):
        self.data2idx([self.train_file, self.valid_file, self.test_file])
        self.load_e_r_idx()
