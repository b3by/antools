import tensorflow as tf


def generate_weight(shape, name='W'):
    """Generate a weight variable

    This method generates weights of the specified shape. Weights are
    initialized with a truncated normal with 0.1 as standard deviation.
    """

    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def generate_bias(shape, name='b'):
    """Generate a bias variable

    This method generates biases of the specified shape. Biases are all
    initialized with a 0.1 constant value.
    """

    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)
