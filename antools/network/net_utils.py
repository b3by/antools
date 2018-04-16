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
    has to be provided.

    Arguments:

    x  -- the input of the dense layer
    number_units -- the number of units that compose the layer
    name -- the name of the layer scope
    """

    with tf.name_scope(name):
        s = 1

        for d in x.shape:
            s *= d if type(d) is int and d > 0 else 1

        flattened = tf.reshape(x, [-1, s])
        weights = generate_weights([s, number_units])
        biases = generate_biases([number_units])

        return tf.nn.relu(tf.add(tf.matmul(flattened, weights), biases))
