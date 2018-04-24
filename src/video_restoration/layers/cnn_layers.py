import tensorflow as tf
from video_restoration.utils.tools import variable_summaries


def convolutional_layer(layer_input, layer_size, activation="linear", stride=1, stddev=0.1, padding='same',
                        name="convolutional_layer", reuse=False):
    # Input layer shape
    input_shape = layer_input.get_shape().as_list()[1:]
    input_shape_filters = input_shape[-1]

    with tf.variable_scope(name, reuse=reuse):
        weights = tf.Variable(tf.truncated_normal([layer_size[0], layer_size[1], input_shape_filters,
                                                   layer_size[2]], stddev=stddev))
        bias = tf.Variable(tf.ones([layer_size[2]]) / 10)
        not_activated = tf.nn.conv2d(layer_input, weights, strides=[1, stride, stride, 1], padding=padding.upper())
        not_activated = tf.nn.bias_add(not_activated, bias)

        if activation != 'linear':
            activated_output = getattr(tf.nn, activation)(not_activated)
        else:
            activated_output = not_activated

        #  Get histograms
        variable_summaries(weights, "weights")
        variable_summaries(bias, "biases")
        tf.summary.histogram("activations", activated_output)
    return activated_output


def convolutional1d_layer(layer_input, layer_size, activation="linear", stride=1, stddev=0.1, padding='valid',
                          name="convolutional1d_layer"):
    # Input layer shape
    shape_in = layer_input.get_shape().as_list()
    layer_size = [shape_in[1]] + layer_size
    conv_out = convolutional_layer(layer_input, layer_size, activation, stride, stddev, padding, name)
    return conv_out


def convolutional3d_layer(layer_input, layer_size, activation="linear", stride=1, stride_d=1, stddev=0.1,
                          padding='same', name="convolutional3d_layer"):
    input_shape = layer_input.get_shape().as_list()[1:]
    input_shape_filters = input_shape[-1]
    with tf.variable_scope(name):
        # Define weights
        weights = tf.get_variable("weights", [layer_size[0], layer_size[1], layer_size[2],
                                              input_shape_filters, layer_size[3]],
                                  initializer=get_initializer_by_name("xavier", stddev=stddev),
                                  dtype=tf.float32, trainable=True)
        not_activated = tf.nn.conv3d(layer_input, weights,
                                     strides=[1, stride, stride, stride_d, 1],
                                     padding=padding.upper())

        # Define biases
        bias = tf.get_variable("biases", [layer_size[3]], initializer=tf.constant_initializer(0.1),
                               dtype=tf.float32, trainable=True)
        not_activated = tf.nn.bias_add(not_activated, bias)

        # Use activation
        if activation != 'linear':
            activated_output = getattr(tf.nn, activation)(not_activated)
        else:
            activated_output = not_activated

        variable_summaries(weights, "weights")
        variable_summaries(bias, "biases")
        tf.summary.histogram("activations", activated_output)
        return activated_output


def get_initializer_by_name(initializer_name, stddev=0.1):
    if initializer_name == "zeros":
        return tf.zeros_initializer()
    elif initializer_name == "normal":
        return tf.random_normal_initializer(stddev=stddev)
    elif initializer_name == "xavier":
        return tf.contrib.layers.xavier_initializer()