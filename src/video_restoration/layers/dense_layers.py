import numpy as np
import tensorflow as tf
from video_restoration.utils.tools import variable_summaries


def faltten(layer_input, name="flatten"):
    with tf.name_scope(name):
        input_shape = layer_input.get_shape().as_list()[1:]
    return tf.reshape(layer_input, shape=[-1, np.prod(input_shape)])


def fully_connected_layer(layer_input, output_size, activation="linear", name="fully_connected"):
    input_shape = layer_input.get_shape().as_list()[1:]
    with tf.name_scope(name):
        weights = tf.Variable(tf.truncated_normal([int(np.prod(input_shape)), output_size], stddev=0.1))
        bias = tf.Variable(tf.ones([output_size]) / 10)

        not_activated = tf.matmul(layer_input, weights)
        not_activated = tf.nn.bias_add(not_activated, bias)

        if activation != 'linear':
            activated_output = getattr(tf.nn, activation)(not_activated)
        else:
            activated_output = not_activated
        variable_summaries(weights, "weights")
        variable_summaries(bias, "biases")
        tf.summary.histogram("activations", activated_output)
    return activated_output
