import numpy as np

from model import GraphNeuralNetwork
from optim import calc_grads, SGD
from function import bce_with_logit


def main():
    n = np.random.randint(2, 20)

    # Generate an undirected graph
    graph = np.random.randint(2, size=[n, n])
    graph = np.tril(graph, -1) + np.tril(graph, -1).T

    label = np.random.randint(2)
    print(graph)
    print(graph.shape)
    print('label', label)

    dim_feature = 10
    x = np.zeros([n, dim_feature])
    x[:, 0] = 1
    W = np.random.normal(0, 0.4, [dim_feature, dim_feature])
    A = np.random.normal(0, 0.4, dim_feature)
    b = np.array([0.])
    model = GraphNeuralNetwork(W, A, b, 2)
    optimizer = SGD(model, lr=0.001)
    for i in range(500):
        grads_flat = calc_grads(model, graph, x, label, lossfunc=bce_with_logit, eps=1e-4)

        outputs = model(graph, x)
        train_loss = bce_with_logit(outputs, label)
        optimizer.update(grads_flat)
        print('step: %d, train_loss: %.15f' % (i, train_loss))

if __name__ == '__main__':
    main()
