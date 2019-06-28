import os
import numpy as np


def get_train_data(data_dir):
    dirs = os.listdir(os.path.join(data_dir, 'train'))
    train_data = []
    graph, label = None, None
    for i in range(len(dirs)//2):
        with open(os.path.join(data_dir, 'train', '%d_graph.txt'%i)) as o:
            n = int(o.readline().strip())
            graph = [list(map(int, o.readline().split())) for _ in range(n)]
            graph = np.array(graph)

        with open(os.path.join(data_dir, 'train', '%d_label.txt'%i)) as o:
            label = int(o.readline().strip())
        assert graph is not None
        assert label is not None
        train_data.append((graph, label))

    return train_data


def get_test_data(data_dir):
    dirs = os.listdir(os.path.join(data_dir, 'test'))
    test_data = []
    graph = None
    for i in range(len(dirs)):
        with open(os.path.join(data_dir, 'test', '%d_graph.txt'%i)) as o:
            n = int(o.readline().strip())
            graph = [list(map(int, o.readline().split())) for _ in range(n)]
            graph = np.array(graph)
        assert graph is not None

        test_data.append(graph)

    return test_data


def random_split(dataset, train_ratio):
    n_data = len(dataset)
    n_train = int(train_ratio*n_data)
    ids = np.arange(len(dataset))
    np.random.shuffle(ids)
    train_data = [dataset[j] for j in ids[:n_train]]
    valid_data = [dataset[j] for j in ids[n_train:]]
    return train_data, valid_data


def get_shuffled_batches(dataset, batch_size):
    ids = np.arange(len(dataset))
    np.random.shuffle(ids)
    batches = []
    for i in range(0, len(dataset), batch_size):
        graphs = [dataset[j][0] for j in ids[i : i + batch_size]]
        labels = [dataset[j][1] for j in ids[i : i + batch_size]]
        batches.append((graphs, labels))
    return batches

class AverageMeter:

    def __init__(self):
        self.cnt = 0
        self.avg = 0.

    def update(self, val, n):
        self.avg = (self.avg*self.cnt + val*n)/(self.cnt + n)
        self.cnt += n
