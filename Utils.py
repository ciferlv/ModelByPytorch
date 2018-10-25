import numpy as np
import math
import matplotlib.pyplot as plt


class Util:
    def shuffle_x_y(self, x, y):
        temp = list(zip(np.array(x), np.array(y)))
        np.random.shuffle(temp)
        x_, y_ = zip(*temp)
        return np.array(x_), np.array(y_)

    def split_by_mini_batch(self, mini_batch, x, y):
        x, y = self.shuffle_x_y(x, y)
        x_splited = []
        y_splited = []
        x_num = len(x)
        split_times = math.ceil(x_num / mini_batch)
        for i in range(split_times):
            if i != split_times - 1:
                start = i * mini_batch
                end = (i + 1) * mini_batch
                x_splited.append(x[start:end])
                y_splited.append(y[start:end])
            else:
                start = i * mini_batch
                x_splited.append(x[start:])
                y_splited.append(y[start:])
        return np.array(x_splited), np.array(y_splited)

    def tensor_to_array(self, t):
        return t.detach().numpy()

    def tensor_to_list(self, t, flatten):
        if not flatten:
            return t.detach().tolist()
        else:
            return t.data.storage().tolist()

    def plot_scatter_diagram(self, title, x_label, y_label, x, y):
        plt.scatter(x, y)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.show()

    def plot_normal_histogram(self, x, range):
        plt.hist(x, range=range)
        plt.show()




if __name__ == "__main__":
    util = Util()
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    a_, b_ = util.shuffle_x_y(a, b)
    print(a)
    print(b)
    print(a_)
    print(b_)
    a_splited, b_splited = util.split_by_mini_batch(2, a, b)
    print(a)
    print(b)
    print(a_splited, b_splited)
