import numpy as np

from model import GraphNeuralNetwork
from optim import calc_grads, Adam
from function import bce_with_logit, sigmoid
import util


def main(n_aggregation, dim_feature, n_epochs, batch_size, eps, outputfile):
    W = np.random.normal(0, 0.4, [dim_feature, dim_feature])
    A = np.random.normal(0, 0.4, dim_feature)
    b = np.array([0.])
    model = GraphNeuralNetwork(W, A, b, n_aggregation=n_aggregation)
    optimizer = Adam(model)

    # Training
    train_data = util.get_train_data('../../datasets')
    print('train_size: %d' % len(train_data))
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

        print('epoch: %d, train_loss: %f, train_acc: %f' % (epoch, train_loss.avg, train_acc.avg))

    # Prediction
    test_data = util.get_test_data('../../datasets')
    with open(outputfile, 'w') as o:
        for graph in test_data:
            x = np.zeros([len(graph), dim_feature])
            x[:, 0] = 1
            logit = model(graph, x)
            pred = sigmoid(logit) > 0.5
            o.write(str(int(pred[0])) + '\n')


if __name__ == '__main__':
    main(
        n_aggregation=2,
        dim_feature=8,
        n_epochs=50,
        batch_size=20,
        eps=0.001,
        outputfile='prediction.txt'
    )
