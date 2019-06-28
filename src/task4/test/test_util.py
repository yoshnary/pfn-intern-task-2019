import unittest
import os
import numpy as np

import util


class TestModel(unittest.TestCase):

    def all_samples(self, sample_dir, test_func):
        for filename in os.listdir(sample_dir):
            if filename == 'README.md':
                continue
            print(filename)
            test_func(os.path.join(sample_dir, filename))

    def test_get_train_data(self):
        print('get_train_data')
        data_dir = '../../datasets'
        train_data = util.get_train_data(data_dir)

        self.assertEqual(len(train_data), 2000)
        for p in train_data:
            self.assertEqual(len(p), 2)
            self.assertEqual(len(p[0].shape), 2)
            self.assertEqual(p[0].shape[0], p[0].shape[1])
            self.assertTrue(p[1] in [0, 1])

    def test_get_test_data(self):
        print('get_test_data')
        data_dir = '../../datasets'
        test_data = util.get_test_data(data_dir)

        self.assertEqual(len(test_data), 500)
        for g in test_data:
            self.assertEqual(len(g.shape), 2)
            self.assertEqual(g.shape[0], g.shape[1])

    def test_random_split(self):
        print('random_split')
        n = 2000
        dataset = np.arange(n)
        ratios = [0.01*i for i in range(101)]
        for r in ratios:
            train_data, valid_data = util.random_split(dataset, r)

            self.assertEqual(len(train_data), int(n*r), 'train_ratio %f'%r)
            self.assertCountEqual(np.hstack([train_data, valid_data]), dataset, 'train_ratio %f'%r)

    def test_get_shuffled_batches(self):
        print('get_shuffled_batches')
        n = 2000
        dataset = np.array([(i, i) for i in range(n)])

        for batch_size in range(1, n + 1):
            batches = util.get_shuffled_batches(dataset, batch_size)

            self.assertEqual(len(batches), (n + batch_size - 1)//batch_size, 'batch_size %d'%batch_size)
            for i in range(len(batches) - 1):
                self.assertEqual(len(batches[i][0]), batch_size, 'batch_size %d'%batch_size)
                self.assertEqual(len(batches[i][1]), batch_size, 'batch_size %d'%batch_size)
            if n%batch_size == 0:
                self.assertEqual(len(batches[-1][0]), batch_size, 'batch_size %d'%batch_size)
                self.assertEqual(len(batches[-1][1]), batch_size, 'batch_size %d'%batch_size)
            else:
                self.assertEqual(len(batches[-1][0]), n%batch_size, 'batch_size %d'%batch_size)
                self.assertEqual(len(batches[-1][1]), n%batch_size, 'batch_size %d'%batch_size)

    def test_average_meter(self):
        def test_sample(filename):
            with open(filename, 'r') as o:
                n = int(o.readline().strip())
                meter = util.AverageMeter()

                for i in range(n):
                    val, n, avg = list(map(float, o.readline().split()))
                    meter.update(val, n)

                    self.assertTrue(np.isclose(meter.avg, avg))

        print('average meter')
        self.all_samples(os.path.join('test', 'average_meter'), test_sample)
