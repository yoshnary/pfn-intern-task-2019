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

def linear(x, W, b):
    s = x @ W.T + b
    return s


class GraphNeuralNetwork:

    def __init__(self, W, A, b, T):
        self.W = W
        self.A = A
        self.b = b
        self.T = T

    def __call__(self, graph, x):
        """Forward propagation.

        Args:
            graph (numpy.ndarry): Adjacency matrix of the input graph.
            x (numpy.nudarry): Input vectors.
                Its shape must be (The number of vertices, dimension of the input vector).
        """            
        s = self.forward_with(self.W, self.A, self.b, graph, x)
        return s

    def forward_with(self, W, A, b, graph, x):
        assert graph.shape[0] == graph.shape[1]
        assert graph.shape[0] == x.shape[0]

        h = x
        for _ in range(self.T):
            h = aggregate1(graph, h)
            h = aggregate2(W, h)
            assert x.shape == h.shape

        h = readout(h)
        s = linear(h, A, b)
        return s

    def params(self):
        return (self.W, self.A, self.b)
