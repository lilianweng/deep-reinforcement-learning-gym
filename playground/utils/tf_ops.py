import tensorflow as tf
import numpy as np
from gym.utils import colorize


def dense_nn(inputs, layers_sizes, name="mlp", reuse=False, output_fn=None, dropout_keep_prob=None,
             batch_norm=False, training=True):
    print(colorize("Building mlp {} | sizes: {}".format(
        name, [inputs.shape[0]] + layers_sizes), "green"))

    with tf.variable_scope(name, reuse=reuse):
        out = inputs
        for i, size in enumerate(layers_sizes):
            print("Layer:", name + '_l' + str(i), size)
            if i > 0 and dropout_keep_prob is not None and training:
                # No dropout on the input layer.
                out = tf.nn.dropout(out, dropout_keep_prob)

            out = tf.layers.dense(
                out,
                size,
                # Add relu activation only for internal layers.
                activation=tf.nn.relu if i < len(layers_sizes) - 1 else None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name=name + '_l' + str(i),
                reuse=reuse
            )

            if batch_norm:
                out = tf.layers.batch_normalization(out, training=training)

        if output_fn:
            out = output_fn(out)

    return out


def conv2d_net(inputs, layers_sizes, name="conv2d", conv_layers=2, with_pooling=True,
               dropout_keep_prob=None, training=True):
    print(colorize("Building conv net " + name, "green"))
    print("inputs.shape =", inputs.shape)

    with tf.variable_scope(name):
        for i in range(conv_layers):
            # Apply convolution computation using a kernel of size (5, 5) over the image
            # inputs with strides (2, 2) and 'valid' padding.
            #   For example:
            #       i = <input image size>, k = 5, s = 2, p = k // 2 = 2
            #       o = (i + 2p - k) // 2 + 1 = (i - 1) // 2 + 1
            #   Read more: https://arxiv.org/pdf/1603.07285.pdf
            #              https://github.com/vdumoulin/conv_arithmetic
            # The output tensor of shape (
            #       batch_size,
            #       (input_image_height - 1) // 2 + 1,
            #       (input_image_width - 1) // 2 + 1,
            #       output_dim,
            # ).
            inputs = tf.layers.conv2d(inputs, 32, [5, 5], strides=[2, 2], name='conv' + str(i))
            print('conv' + str(i) + '.shape =', inputs.shape)

            if with_pooling:
                inputs = tf.layers.max_pooling2d(inputs, [2, 2], 2, name='pool' + str(i))
                print('pool' + str(i) + '.shape =', inputs.shape)

        flatten = tf.reshape(inputs, [-1, np.prod(inputs.shape.as_list()[1:])], name='flatten')
        outputs = dense_nn(flatten, layers_sizes, name='fc', dropout_keep_prob=dropout_keep_prob)

        print("flatten.shape =", flatten.shape)
        print("outputs.shape =", outputs.shape)

    return outputs


def alexnet(inputs, output_size, training=True, name='alexnet', dropout_keep_prob=0.5):
    """alex net v2

    Described in: http://arxiv.org/pdf/1404.5997v2.pdf
    Parameters from:
    github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
    layers-imagenet-1gpu.cfg

    Refer to: https://github.com/tensorflow/models/blob/master/research/slim/nets/alexnet.py
    """
    trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

    with tf.variable_scope(name):
        net = tf.layers.conv2d(inputs, 64, [11, 11], 4, padding='valid', name='conv1')
        net = tf.layers.max_pooling2d(net, [3, 3], 2, name='pool1')
        net = tf.layers.conv2d(net, 192, [5, 5], name='conv2')
        net = tf.layers.max_pooling2d(net, [3, 3], 2, name='pool2')
        net = tf.layers.conv2d(net, 384, [3, 3], name='conv3')
        net = tf.layers.conv2d(net, 384, [3, 3], name='conv4')
        net = tf.layers.conv2d(net, 256, [3, 3], name='conv5')
        net = tf.layers.max_pooling2d(net, [3, 3], 2, name='pool5')

        # Use conv2d instead of fully_connected layers.
        net = tf.layers.conv2d(net, 4096, [5, 5], padding='valid', name='fc6',
                               kernel_initializer=trunc_normal(0.005),
                               bias_initializer=tf.constant_initializer(0.1))

        net = tf.layers.dropout(net, dropout_keep_prob, training=training, name='dropout6')
        net = tf.layers.conv2d(net, 4096, [1, 1], name='fc7')

        if output_size:
            net = tf.layers.dropout(net, dropout_keep_prob, training=training, name='dropout7')
            net = tf.layers.conv2d(net, output_size, [1, 1], name='fc8')

    return net


def lstm_net(inputs, layers_sizes, name='lstm', step_size=16, lstm_layers=1, lstm_size=256,
             pre_lstm_dense_layer=None, dropout_keep_prob=None, training=True):
    """inputs = (batch_size * step_size, *observation_size)
    """
    print(colorize("Building lstm net " + name, "green"))
    print("inputs.shape =", inputs.shape)

    state_size = inputs.shape.as_list()[1]
    inputs = tf.reshape(inputs, [-1, step_size, state_size])
    print("reshaped inputs.shape =", inputs.shape)

    def _make_cell():
        cell = tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=True, reuse=not training)
        if training and dropout_keep_prob:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
        return cell

    with tf.variable_scope(name):

        if pre_lstm_dense_layer:
            inputs = tf.nn.relu(dense_nn(inputs, [pre_lstm_dense_layer], name='pre_lstm'))

        with tf.variable_scope('lstm_cells'):
            # Before transpose, inputs.get_shape() = (batch_size, num_steps, lstm_size)
            # After transpose, inputs.get_shape() = (num_steps, batch_size, lstm_size)
            lstm_inputs = tf.transpose(inputs, [1, 0, 2])

            cell = tf.contrib.rnn.MultiRNNCell([
                _make_cell() for _ in range(lstm_layers)], state_is_tuple=True)
            lstm_outputs, lstm_states = tf.nn.dynamic_rnn(cell, lstm_inputs, dtype=tf.float32)

            # transpose back.
            lstm_outputs = tf.transpose(lstm_outputs, [1, 0, 2])

            print("cell =", cell)
            print("lstm_states =", lstm_states)
            print("lstm_outputs.shape =", lstm_outputs.shape)

        outputs = dense_nn(lstm_outputs, layers_sizes, name="outputs")
        print("outputs.shape =", outputs.shape)

        outputs = tf.reshape(outputs, [-1, layers_sizes[-1]])
        print("reshaped outputs.shape =", outputs.shape)
        return outputs
