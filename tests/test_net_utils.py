import unittest
import types

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
        test_x = tf.placeholder(tf.float32, shape=[None, 2, 3, 4])
        layer = net_utils.dense_layer(test_x, 233, 'test_dense')

        self.assertEqual(layer.shape[1], 233)
        self.assertEqual(layer.name, 'test_dense/Relu:0')

    def test_dropout(self):
        test_x = tf.placeholder(tf.float32, shape=[None, 100])
        test_prob = tf.placeholder_with_default(-.5, shape=(), name='keep_tst')
        layer = net_utils.dense_layer(test_x, 120, 'test_drop_in')
        drp = net_utils.drop_layer(layer, test_prob, 't_drop')

        # only to check that the dropout does not modify the input shape
        self.assertEqual(drp.shape[1], 120)
        self.assertEqual(drp.name, 't_drop/dropout/mul:0')

    def test_dense_layers(self):
        test_units = [37, 22, 18]
        test_x = tf.placeholder(tf.float32, shape=[None, 8, 7, 9])
        test_prob = tf.placeholder_with_default(-.5, shape=(), name='keep_tst')
        layers = net_utils.dense_layers(test_x, test_units, test_prob,
                                        'test_d_lay')

        self.assertEqual(len(layers), 6)
        self.assertEqual(layers[0].shape[1], 37)
        self.assertEqual(layers[1].shape[1], 37)
        self.assertEqual(layers[2].shape[1], 22)
        self.assertEqual(layers[3].shape[1], 22)
        self.assertEqual(layers[4].shape[1], 18)
        self.assertEqual(layers[5].shape[1], 18)

    def test_depth_conv2d_layer(self):
        test_x = tf.placeholder(tf.float32, shape=[None, 10, 10, 3])
        test_k = types.SimpleNamespace(kernel=[3, 3], strides=[1, 1, 1, 1],
                                       depth=2)
        conv = net_utils.depth_conv2d_layer(test_x, test_k, 'conv')

        self.assertEqual(conv.shape[1].value, 10)
        self.assertEqual(conv.shape[2].value, 10)
        self.assertEqual(conv.shape[3].value, 6)

    def test_depth_conv2d_layer_valid_padding(self):
        test_x = tf.placeholder(tf.float32, shape=[None, 10, 10, 3])
        test_k = types.SimpleNamespace(kernel=[3, 3], strides=[1, 1, 1, 1],
                                       depth=2)
        conv = net_utils.depth_conv2d_layer(test_x, test_k, 'conv',
                                            padding='VALID')

        self.assertEqual(conv.shape[1], 8)
        self.assertEqual(conv.shape[2], 8)
        self.assertEqual(conv.shape[3], 6)

    def test_maxpool_layer(self):
        test_x = tf.placeholder(tf.float32, shape=[None, 10, 10, 3])
        test_k = types.SimpleNamespace(kernel=[1, 2, 2, 1],
                                       strides=[1, 2, 2, 1])
        m_pool = net_utils.maxpool_layer(test_x, test_k, 'test_pool')

        self.assertEqual(m_pool.shape[1], 5)
        self.assertEqual(m_pool.shape[2], 5)

    def test_maxpool_layer_same_padding(self):
        test_x = tf.placeholder(tf.float32, shape=[None, 12, 12, 3])
        test_k = types.SimpleNamespace(kernel=[1, 3, 3, 1],
                                       strides=[1, 2, 2, 1])
        m_pool = net_utils.maxpool_layer(test_x, test_k, 'test_pool',
                                         padding='SAME')

        self.assertEqual(m_pool.shape[1], 6)
        self.assertEqual(m_pool.shape[2], 6)

    def test_softmax_layer(self):
        test_x = tf.placeholder(tf.float32, shape=[None, 100])
        test_soft = net_utils.softmax_layer(test_x, 3, 'test_softmax')

        self.assertEqual(test_soft.shape[1], 3)

    def test_cross_entropy(self):
        # should be implemented!
        pass


if __name__ == '__main__':
    unittest.main()
