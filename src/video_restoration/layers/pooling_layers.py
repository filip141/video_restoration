import tensorflow as tf


def maxpool_layer(layer_input, pool_size, stride=1, padding='same', name="maxpool_layer"):
    # Input layer shape
    with tf.name_scope(name):
        k_size = [1, ] + pool_size + [1, ]
        layer_output = tf.nn.max_pool(layer_input, ksize=k_size, strides=[1, stride, stride, 1],
                                      padding=padding.upper())

        #  Get histograms
        tf.summary.histogram("max_pooling_histogram", layer_output)
    return layer_output


def maxpool3d_layer(layer_input, pool_size, stride=1, padding='same', name="maxpool3d_layer"):
    # Input layer shape
    with tf.name_scope(name):
        k_size = [1, ] + pool_size + [1, ]
        layer_output = tf.nn.max_pool3d(layer_input, ksize=k_size,
                                        strides=[1, stride, stride, stride, 1],
                                        padding=padding.upper())

        #  Get histograms
        tf.summary.histogram("max_pooling_histogram", layer_output)
    return layer_output


