import unittest
import os
import numpy as np

import optim


class TestModel(unittest.TestCase):

    def all_samples(self, sample_dir, test_func):
        for filename in os.listdir(sample_dir):
            if filename == 'README.md':
                continue
            print(filename)
            test_func(os.path.join(sample_dir, filename))

    def test_calc_grads(self):
        class TestModel:
            def __init__(self, W, b):
                self.W = W
                self.b = b

            def __call__(self, graph, x):
                return self.forward_with(self.W, self.b, graph, x)

            def forward_with(self, W, b, graph, x):
                u = W * graph
                v = b * x
                w = W * b
                return np.hstack([u, v, w])

            def params(self):
                return (self.W, self.b)

        def test_loss(x, *args):
            return x.sum()

        def test_sample(filename):
            with open(filename, 'r') as o:
                n = int(o.readline().strip())
                W = np.zeros([n])
                b = np.zeros([n])
                G = np.zeros([n])
                x = np.zeros([n])
                ans = np.zeros([2*n])

                W[:] = list(map(float, o.readline().split()))
                b[:] = list(map(float, o.readline().split()))
                G[:] = list(map(float, o.readline().split()))
                x[:] = list(map(float, o.readline().split()))
                ans[:] = list(map(float, o.readline().split()))

            model = TestModel(W, b)
            grads_flat = optim.calc_grads(model, G, x, None, test_loss, eps=1e-4)

            self.assertTrue(np.isclose(grads_flat, ans).all(), filename)

        print('calc_grads')
        self.all_samples(os.path.join('test', 'calc_grads'), test_sample)

    def test_adam(self):
        class TestModel:
            def __init__(self, W, b):
                self.W = W
                self.b = b

            def params(self):
                return (self.W, self.b)

        def test_sample(filename):
            with open(filename, 'r') as o:
                lr = float(o.readline().strip())
                p, q, r = list(map(int, o.readline().split()))
                W = np.zeros([p, q])
                b = np.zeros([r])

                for i in range(p):
                    line = list(map(float, o.readline().split()))
                    W[i, :] = line
                line = list(map(float, o.readline().split()))
                b[:] = line

                n = int(o.readline().strip())
                grads_flat = np.zeros([n, p*q + r])
                W_ans = np.zeros([p, q])
                b_ans = np.zeros([r])
                m_ans = np.zeros([p*q + r])
                v_ans = np.zeros([p*q + r])

                for i in range(n):
                    line = list(map(float, o.readline().split()))
                    grads_flat[i, :] = line
                for i in range(p):
                    line = list(map(float, o.readline().split()))
                    W_ans[i, :] = line
                line = list(map(float, o.readline().split()))
                b_ans[:] = line
                line = list(map(float, o.readline().split()))
                m_ans[:] = line
                line = list(map(float, o.readline().split()))
                v_ans[:] = line

            model = TestModel(W, b)
            optimizer = optim.Adam(model, lr)
            for i in range(n):
                optimizer.update(grads_flat[i])

            self.assertTrue(np.isclose(model.params()[0], W_ans).all(), filename)
            self.assertTrue(np.isclose(model.params()[1], b_ans).all(), filename)
            self.assertTrue(np.isclose(optimizer.m, m_ans).all(), filename)
            self.assertTrue(np.isclose(optimizer.v, v_ans).all(), filename)

        print('adam')
        self.all_samples(os.path.join('test', 'adam'), test_sample)
