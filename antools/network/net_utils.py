from functools import reduce

import tensorflow as tf


def generate_weights(shape, name='W'):
    """Generate a weight variable

    This method generates weights of the specified shape. Weights are
    initialized with a truncated normal with 0.1 as standard deviation.
    """

    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def generate_biases(shape, name='b'):
    """Generate a bias variable

    This method generates biases of the specified shape. Biases are all
    initialized with a 0.1 constant value.
    """

    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def dense_layer(x, number_units, name):
    """Generate a dense layer

    This method creates a dense layer with as many units as specified in the
    number_units argument. The input of the layer is x. The name of the scope
    has to be provided. The input x will be reshaped to a flat tensor, where
    the batch size is left undefined.

    Arguments:

    x  -- the input of the dense layer
    number_units -- the number of units that compose the layer
    name -- the name of the layer scope
    """

    with tf.name_scope(name):
        s = reduce(lambda x, y: x * y, x.get_shape().as_list()[1:])
        flattened = tf.reshape(x, [-1, s])
        weights = generate_weights([flattened.shape[1].value, number_units])
        biases = generate_biases([number_units])

        return tf.nn.relu(tf.add(tf.matmul(flattened, weights), biases))


def drop_layer(x, keep_probability, name):
    """Generate a dropout layer

    This method generates a dropout layer that is applied to the input. The
    probability of keeping the neurons in the input layer and the name of
    the layer scope have to be specified as arguments.

    Arguments:

    x -- the input of the dropout layer
    keep_probability -- the probability of keeping neurons (a tensor)
    name -- the name of the layer

    """

    with tf.name_scope(name):
        return tf.nn.dropout(x, keep_probability)


def dense_layers(x, units, keep_probability, name_prefix):
    """Generate a stack of dense layers

    This method generates a stack of fully connected layers. The layers will
    take x as input, and each layer will contain as many units as the
    corresponding value in the units list. The prefix of the scope name has to
    be provided. Each dense layer is interleaved with a dropout layer.

    Arguments:

    x -- the input of the dense layer
    units -- a list containing the units to be included in each layer
    keep_probability -- the probability of keeping units in the layers
    name_prefix -- the prefix for the scope name of each dense layer

    """

    layers = []

    for idx, u in enumerate(units):
        nm = name_prefix + '{}'.format(idx + 1)
        in_layer = x if idx == 0 else layers[-1]
        f = dense_layer(in_layer, u, nm)
        layers.append(f)
        drop = drop_layer(f, keep_probability, 'drop{}'.format(idx + 1))
        layers.append(drop)

    return layers


def depth_conv2d_layer(x, kernel, name, padding='SAME'):
    """Generate a depthwise convolutional layer

    This method produces a layer by applying depthwise, 2D convolution to
    the input. The kernel should contain 2 dimensions only.

    Arguments:

    x -- the input of the convolution
    kernel -- namespace with 'kernel' and 'depth' fields
    name -- the name of the layer
    padding -- the type of padding to apply (default SAME)
    """
    with tf.name_scope(name):
        channels = x.shape[3].value
        depth = kernel.depth
        k = kernel.kernel
        strides = kernel.strides

        w = generate_weights([k[0], k[1], channels, depth])
        b = generate_biases([channels * depth])

        conv = tf.nn.depthwise_conv2d(x, w, strides, padding=padding)
        act = tf.nn.relu(tf.add(conv, b))

        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activations', act)

        return act
