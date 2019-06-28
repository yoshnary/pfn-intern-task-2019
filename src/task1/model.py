import numpy as np


def aggregate1(graph, x):
    assert graph.shape[0] == graph.shape[1]
    
    a = graph @ x
    return a

def aggregate2(W, x):
    assert W.shape[0] == W.shape[1]
    
    x = np.maximum(0, x @ W.T)
    return x

def readout(x):
    h = x.sum(axis=0)
    return h


class GraphNeuralNetwork:

    def __init__(self, W, n_aggregation):
        self.W = W
        self.n_aggregation = n_aggregation

    def __call__(self, graph, x):
        """Forward propagation.

        Args:
            graph (numpy.ndarry): Adjacency matrix of the input graph.
            x (numpy.nudarry): Input vectors.
                Its shape must be (The number of vertices, dimension of the input vector).
        """
        h = x
        for _ in range(self.n_aggregation):
            h = aggregate1(graph, h)
            h = aggregate2(self.W, h)
            assert h.shape == x.shape
        h = readout(h)
        return h
