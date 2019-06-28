import numpy as np

from model import GraphNeuralNetwork
from optim import calc_grads, MomentumSGD
from function import bce_with_logit, sigmoid
import util


def test(model, dataset, dim_feature):
    """Function for model evaluation"""
    acc = util.AverageMeter()
    loss = util.AverageMeter()
    for graph, label in dataset:
        x = np.zeros([len(graph), dim_feature])
        x[:, 0] = 1
        outputs = model(graph, x)
        loss.update(bce_with_logit(outputs, label), 1)
        acc.update((sigmoid(outputs) > 0.5) == label, 1)
    return loss.avg, acc.avg


def main(n_aggregation, lr, momentum, dim_feature, n_epochs, batch_size, eps):
    W = np.random.normal(0, 0.4, [dim_feature, dim_feature])
    A = np.random.normal(0, 0.4, dim_feature)
    b = np.array([0.])
    model = GraphNeuralNetwork(W, A, b, n_aggregation=n_aggregation)
    optimizer = MomentumSGD(model, lr, momentum)

    dataset = util.get_train_data('../../datasets')
    train_data, valid_data = util.random_split(dataset, train_ratio=0.5)
    print('train_size: %d, valid_size: %d' % (len(train_data), len(valid_data)))

    for epoch in range(n_epochs):
        train_loss = util.AverageMeter()
        train_acc = util.AverageMeter()
        for graphs, labels in util.get_shuffled_batches(train_data, batch_size):
            grads_flat = 0
            for graph, label in zip(graphs, labels):
                x = np.zeros([len(graph), dim_feature])
                x[:, 0] = 1
                grads_flat += calc_grads(model, graph, x, label, bce_with_logit, eps)/batch_size

                outputs = model(graph, x)
                train_loss.update(bce_with_logit(outputs, label), 1)
                train_acc.update((sigmoid(outputs) > 0.5) == label, 1)

            optimizer.update(grads_flat)

        valid_loss, valid_acc = test(model, valid_data, dim_feature)
        print('epoch: %d, train_loss: %f, train_acc: %f, valid_loss: %f, vald_acc: %f'
                % (epoch, train_loss.avg, train_acc.avg, valid_loss, valid_acc))


if __name__ == '__main__':
    main(
        n_aggregation=2,
        lr=0.001,
        momentum=0.9,
        dim_feature=8,
        n_epochs=100,
        batch_size=20,
        eps=0.001
    )
