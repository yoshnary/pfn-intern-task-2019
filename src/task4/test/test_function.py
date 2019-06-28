import unittest
import os
import numpy as np

import function


class TestModel(unittest.TestCase):

    def all_samples(self, sample_dir, test_func):
        for filename in os.listdir(sample_dir):
            if filename == 'README.md':
                continue
            print(filename)
            test_func(os.path.join(sample_dir, filename))

    def test_bce_with_logit(self):
        def test_sample(filename):
            with open(filename, 'r') as o:
                logit = float(o.readline().strip())
                label = int(o.readline().strip())
                ans = float(o.readline().strip())

            self.assertTrue(np.isclose(function.bce_with_logit(logit, label), ans).all(), filename)

        print('bce_with_logit')
        self.all_samples(os.path.join('test', 'bce_with_logit'), test_sample)

    def test_sigmoid(self):
        def test_sample(filename):
            with open(filename, 'r') as o:
                logit = float(o.readline().strip())
                ans = float(o.readline().strip())

            self.assertTrue(np.isclose(function.sigmoid(logit), ans).all(), filename)

        print('sigmoid')
        self.all_samples(os.path.join('test', 'sigmoid'), test_sample)
