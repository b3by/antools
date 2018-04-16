import unittest

import tensorflow as tf
from antools.network import net_utils


class TestNetworkUtils(unittest.TestCase):

    def test_generate_weights(self):
        w = net_utils.generate_weights([2, 3])
        self.assertEqual(w.shape[0], 2)
        self.assertEqual(w.shape[1], 3)

    def test_generate_weights_with_name(self):
        w = net_utils.generate_weights([1, 1], name='test_weights')
        self.assertEqual(w.name, 'test_weights:0')

    def test_generate_biases(self):
        b = net_utils.generate_biases([10])
        self.assertEqual(b.shape[0], 10)

    def test_generate_biases_with_name(self):
        b = net_utils.generate_biases([10], name='test_biases')
        self.assertEqual(b.name, 'test_biases:0')

    def test_dense_layer(self):
        test_x = tf.placeholder(tf.float32, shape=[10, 10, 100])
        layer = net_utils.dense_layer(test_x, 100, 'test_dense')
        self.assertEqual(layer.shape[0], 10000)
        self.assertEqual(layer.shape[1], 100)
        self.assertEqual(layer.name, 'test_dense/Relu:0')


if __name__ == '__main__':
    unittest.main()
