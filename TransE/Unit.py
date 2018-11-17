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





if __name__ == "__main__":
    print("dfsd")