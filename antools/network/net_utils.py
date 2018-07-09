"""Tensorflow utilities

This module provides utilities in support of building neural networks. Some
abstractions for layer generation are provided, however, full models should
be built with a bit of care.
"""
from functools import reduce

import tensorflow as tf
from tensorflow.python.ops import math_ops


activations = {
    'relu': tf.nn.relu,
    'sigmoid': tf.nn.sigmoid,
    'tanh': tf.nn.tanh}
"""Activation functions to use

This dictionary contains a set of activation functions that can be used when
building a layer of any kind. Please be aware that some method signatures may
miss required arguments when using particular activation function. This should
be addressed properly in the future, when more activation functions will be
supported.
"""


def generate_weights(shape, name='W'):
    """Generate a weight variable

    This method generates weights of the specified shape. Weights are
    initialized with a truncated normal with 0.1 as standard deviation.

    Parameters
    ----------
    shape : tuple
        A tuple containing all the shapes for the weights.
    name : str
        The name of the weight variable to generate.

    Returns
    -------
    tf.Variable
        A tensorflow variable.
    """
    initial = tf.truncated_normal(shape, stddev=0.2)
    return tf.Variable(initial, name=name)


def generate_biases(shape, name='b'):
    """Generate a bias variable

    This method generates biases of the specified shape. Biases are all
    initialized with a 0.1 constant value.

    Parameters
    ----------
    shape : tuple
        A tuple containing all the shapes for the biases.
    name : str
        The name of the bias variable to generate.

    Returns
    -------
    tf.Variable
        A tensorflow variable.
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def dense_layer(x, number_units, name, activation='relu'):
    """Generate a dense layer

    This method creates a dense layer with as many units as specified in the
    number_units argument. The input of the layer is x. The name of the scope
    has to be provided. The input x will be reshaped to a flat tensor, where
    the batch size is left undefined.

    Parameters
    ----------
    x : tf.Tensor
        The input of the dense layer.
    number_units : int
        The number of units that compose the layer.
    name : str
        The name of the layer scope.
    activation :str
        The activation function to use (optional, default to relu).

    Returns
    -------
    tf.Tensor
        The sum of the input multiplied by the weights and the biases,
        processed with the activation function.
    """
    with tf.name_scope(name):
        s = reduce(lambda x, y: x * y, x.get_shape().as_list()[1:])
        flattened = tf.reshape(x, [-1, s])
        weights = generate_weights([flattened.shape[1].value, number_units])
        biases = generate_biases([number_units])

        return activations[activation](
            tf.add(tf.matmul(flattened, weights), biases))


def drop_layer(x, keep_probability, name):
    """Generate a dropout layer

    This method generates a dropout layer that is applied to the input. The
    probability of keeping the neurons in the input layer and the name of
    the layer scope have to be specified as arguments.

    Parameters
    ----------
    x : tf.Tensor
        The input of the dropout layer.
    keep_probability : tf.Tensor
        The probability of keeping neurons, usually around 0.5 or something.
    name : str
        The name of the layer scope.

    Returns
    -------
    tf.Tensor
        The dropped version of the input layer.
    """
    with tf.name_scope(name):
        return tf.nn.dropout(x, keep_probability)


def dense_layers(x, units, keep_probability, name_prefix, activation='relu'):
    """Generate a stack of dense layers

    This method generates a stack of fully connected layers. The layers will
    take x as input, and each layer will contain as many units as the
    corresponding value in the units list. The prefix of the scope name has to
    be provided. Each dense layer is interleaved with a dropout layer.

    Parameters
    ----------
    x : tf.Tensor
        The input of the dense layer.
    units : tuple
        A list containing the units to be included in each layer.
    keep_probability : tf.Tensor
        The probability of keeping units in the layers.
    name_prefix : str
        The prefix for the scope name of each dense layer.
    activation : str
        The activation function to use (optional, default to relu).

    Returns
    -------
    list
        A list of all the layers produced. The final layer can be accessed with
        layers[-1], and represents hte output of the layer concatenation.
    """
    layers = []

    for idx, u in enumerate(units):
        nm = name_prefix + '{}'.format(idx + 1)
        in_layer = x if idx == 0 else layers[-1]
        f = dense_layer(in_layer, u, nm, activation)
        layers.append(f)
        drop = drop_layer(f, keep_probability, 'drop{}'.format(idx + 1))
        layers.append(drop)

    return layers


def depth_conv2d_layer(x, kernel, name, padding='SAME'):
    """Generate a depthwise convolutional layer

    This method produces a layer by applying depthwise, 2D convolution to
    the input. The kernel should contain 2 dimensions only.

    Parameters
    ----------
    x : tf.Tensor
        The input of the convolution.
    kernel : types.SimpleNamespace
        Namespace with 'kernel', 'strides' and 'depth' fields.
    name : str
        The name of the layer.
    padding : str
        The type of padding to apply (default SAME).

    Returns
    -------
    tf.Tensor
        The output tensor from the generated convolutional layer.
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


def maxpool_layer(x, kernel, name, padding='VALID'):
    """Generate a max pooling layer

    This method can be used to generate a max pooling layer. The input and the
    kernel dimensions should be provided as input. By default, no padding is
    applied to the input layer before the max pooling operation.

    Parameters
    ----------
    x : tf.Tensor
        The input tensor.
    kernel : types.SimpleNamespace
        Namespace with 'kernel' and 'strides' fields.
    name : str
        The name of the layer scope.
    padding : str
        The type of padding to apply to the input (default VALID).

    Returns
    -------
    tf.Tensor
        The input tensor processed with the max pooling layer.
    """
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=kernel.kernel, strides=kernel.strides,
                              padding=padding)


def softmax_layer(x, number_labels, name):
    """Generate softmax layer

    This method generates a softmax layer from the input x, according with the
    number of labels provided as argument.

    Parameters
    ----------
    x : tf.Tensor
        The input tensor.
    number_labels : int
        How many labels should be used.
    name : str
        The name of the layer scope.

    Returns
    -------
    tf.Tensor
        The softmax calculation of the input tensor.
    """
    with tf.name_scope(name):
        weights = generate_weights([x.shape[1].value, number_labels])
        biases = generate_biases([number_labels])

        return tf.nn.softmax(tf.matmul(x, weights) + biases)


def cross_entropy_loss(logits, labels, name, regularize=False):
    """Calculate corss entropy

    This method calculates the cross entropy for a vector of predictions.
    If required, the loss function is calculated with L2 regularization.

    Parameters
    ----------
    logits : tf.Tensor
        The tensor of predicted values.
    labels : tf.Tensor the tensor of ground truth for the points
    name : str
        The scope name of the loss calculation.
    regularize : bool
        Whether L2 regularization should be used (defaulted to False).

    Returns
    ------
    tf.Tensor
        A tensor containing the loss value.
    """
    with tf.name_scope(name):
        xentropy = -tf.reduce_sum(labels * tf.log(logits),
                                  reduction_indices=[1])

        loss = tf.reduce_mean(xentropy)

        if regularize:
            loss = tf.nn.l2_loss(loss)

        return loss


def soft_cross_entropy_loss(last_layer, labels, name):
    """Calculate cross entropy after softmax

    This method returns the cross entropy loss computed over a bunch of logits.

    Parameters
    ----------
    last_layer : tf.Tensor
        The tensor of the unnormalized log probabilities.
    labels : tf.Tensor
        The tensor of the ground truth.
    name : str
        The scope name.

    Returns
    ------
    tf.Tensor
        The tensor containing the computed cross entropy.
    """
    label_number = labels.shape[1].value
    with tf.name_scope(name):
        weights = generate_weights([last_layer.shape[1].value, label_number])
        biases = generate_biases([label_number])
        logits = tf.matmul(last_layer, weights) + biases
        xentropy = tf.softmax_cross_entropy(logits=logits, labels=labels)
        return xentropy


def adam_backprop(loss, learning_rate, global_step, name):
    """Run backpropagation step

    This method runs the backpropagation step with the Adam optimizer.

    Parameters
    ----------
    loss : tf.Tensor
        The loss value to use for the optimization.
    learning_rate : tf.Tensor
        The initial learning rate to use.
    name : str
        The scope name for the optimizer.

    Returns
    -------
    tf.Tensor
        The tensor operation for the graph update.
    """
    with tf.name_scope(name):
        beta1 = 0.3
        beta2 = 0.7
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate, epsilon=0.1, beta1=0.3, beta2=0.7)

        beta1_power = tf.pow(beta1, tf.cast(global_step, tf.float32))
        beta2_power = tf.pow(beta2, tf.cast(global_step, tf.float32))
        lr = (optimizer._lr * math_ops.sqrt(1.0 - beta2_power) / (1.0 -
                                                                  beta1_power))
        tf.summary.scalar('learning_rate', lr)

        return optimizer.minimize(loss, global_step=global_step)


def batchnorm_layer(x, n_out, is_train, name):
    """Batch normalization layer

    This method generate a batch normalization layer.

    Parameters
    ----------
    x : tf.Tensor
        The layer input.
    n_out : int
        Number of output units.
    is_train : tf.Tensor
        Whether the network in in training or testing mode.
    name : str
        The name of the layer.

    Returns
    -------
    tf.Tensor
        The normalized version of the input layer, according to whether the
        network is training or not.
    """
    with tf.name_scope(name):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta',
                           trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma',
                            trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_train, mean_var_with_update, lambda:
                            (ema.average(batch_mean), ema.average(batch_var)))

        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        tf.summary.histogram('norm_activations', normed)

        return normed


def freeze_model(checkpoint_name, model_name, destination, node_names,
                 as_text=False):
    """Freeze a model into a single, servable file

    This method can be used to freeze a previously trained model, from its
    checkpoint folder. A list of nodes needs to be specified.

    Parameters
    ----------
    checkpoint_location : str
        The name of the model checkpoint (exclude the suffix).
    model_name : str
        The name of the file that will contain the model.
    destination : str
        The destination path of the model.
    node_names : list
        A list of nodes that should be available in the graph.
    as_text : bool
        Whether the model should be plain text (defaulted to False). If True,
        the generated model will come in a JSON format.
    """
    session = tf.Session()

    saver = tf.train.import_meta_graph(checkpoint_name + '.meta')
    saver.restore(session, checkpoint_name)

    graph_def = session.graph.as_graph_def()

    # this is needed in case batchnorm is used in the model
    # see https://github.com/tensorflow/tensorflow/issues/3628
    for node in graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr:
                del node.attr['use_locking']

    conv = tf.graph_util.convert_variables_to_constants(session, graph_def,
                                                        node_names)

    tf.train.write_graph(conv, destination, model_name, as_text=as_text)


def unfreeze_model(model_file, x_name, y_name, others, name):
    """Unfreeze a model and get it ready for inference

    This method takes the location of a freezed model, and reads it. In case
    other values should be passed to the graph, a list of tensor names can be
    provided.

    Parameters
    ----------
    model_file : str
        The filename of the model.
    x_name : str
        The name of the input tensor.
    y_name : str
        The name of the output tensor.
    others : list
        A list of names that should be fetched from the graph.
    name : str
        The name of the graph that is produced.

    Returns
    -------
    tuple
        A tuple containing the input node as first element, the output node as
        second element, a dictionary of tensors as third element, and the
        unrolled graph as fourth element.
    """
    with tf.gfile.GFile(model_file, 'rb') as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(restored_graph_def, input_map=None, name=name)

    x = graph.get_tensor_by_name(x_name)
    y = graph.get_tensor_by_name(y_name)
    tensors = {}

    for tensor_name in others:
        tensor = graph.get_tensor_by_name(tensor_name)
        tensors[tensor_name] = tensor

    return x, y, tensors, graph
