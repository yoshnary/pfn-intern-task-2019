import unittest
import os
import numpy as np

import model


class TestModel(unittest.TestCase):

    def all_samples(self, sample_dir, test_func):
        for filename in os.listdir(sample_dir):
            if filename == 'README.md':
                continue
            print(filename)
            test_func(os.path.join(sample_dir, filename))

    def test_aggregate1(self):
        def test_sample(filename):
            with open(filename, 'r') as o:
                n, d = list(map(int, o.readline().split()))
                graph = np.zeros([n, n])
                x = np.zeros([n, d])
                ans = np.zeros([n, d])
                
                for i in range(n):
                    line = list(map(float, o.readline().split()))
                    graph[i, :] = line
                for i in range(n):
                    line = list(map(float, o.readline().split()))
                    x[i, :] = line
                for i in range(n):
                    line = list(map(float, o.readline().split()))
                    ans[i, :] = line

            self.assertTrue(np.isclose(model.aggregate1(graph, x), ans).all(), filename)

        print('aggregate1')
        self.all_samples(os.path.join('test', 'aggregate1'), test_sample)

    def test_aggregate2(self):
        def test_sample(filename):
            with open(filename, 'r') as o:
                n, d = list(map(int, o.readline().split()))
                W = np.zeros([d, d])
                a = np.zeros([n, d])
                ans = np.zeros([n, d])

                for i in range(d):
                    line = list(map(float, o.readline().split()))
                    W[i, :] = line
                for i in range(n):
                    line = list(map(float, o.readline().split()))
                    a[i, :] = line
                for i in range(n):
                    line = list(map(float, o.readline().split()))
                    ans[i, :] = line

            self.assertTrue(np.isclose(model.aggregate2(W, a), ans).all(), filename)

        print('aggregate2')
        self.all_samples(os.path.join('test', 'aggregate2'), test_sample)

    def test_readout(self):
        def test_sample(filename):
            with open(filename, 'r') as o:
                n, d = list(map(int, o.readline().split()))
                x = np.zeros([n, d])
                ans = np.zeros([d])

                for i in range(n):
                    line = list(map(float, o.readline().split()))
                    x[i, :] = line
                line = list(map(float, o.readline().split()))
                ans[:] = line

            self.assertTrue(np.isclose(model.readout(x), ans).all(), filename)

        print('readout')
        self.all_samples(os.path.join('test', 'readout'), test_sample)

    def test_gnn_call(self):
        def test_sample(filename):
            with open(filename, 'r') as o:
                t, n, d = list(map(int, o.readline().split()))
                W = np.zeros([d, d])
                G = np.zeros([n, n])
                x = np.zeros([n, d])
                ans = np.zeros([d])

                for i in range(d):
                    line = list(map(float, o.readline().split()))
                    W[i, :] = line
                for i in range(n):
                    line = list(map(float, o.readline().split()))
                    G[i, :] = line
                for i in range(n):
                    line = list(map(float, o.readline().split()))
                    x[i, :] = line
                line = list(map(float, o.readline().split()))
                ans[:] = line

            gnn = model.GraphNeuralNetwork(W, t)
            self.assertTrue(np.isclose(gnn(G, x), ans).all(), filename)

        print('gnn_call')
        self.all_samples(os.path.join('test', 'gnn_call'), test_sample)
