import tensorflow as tf


def lstm(layer_input, rnn_size, dropout_prob, batch_size, preprocess_data=True, init_state=None, traspose_dims=None,
         name="lstm"):
    if preprocess_data:
        with tf.variable_scope("{}_Preprocess".format(name)):
            pre_shape = [x for x in layer_input.get_shape().as_list() if x != 1]
            input_shape = [-1, *pre_shape[1:]]
            in_reshaped = tf.reshape(layer_input, input_shape)
            # unstacked_data = tf.unstack(in_reshaped, axis=1)
    else:
        in_reshaped = layer_input

    with tf.variable_scope(name):
        rnn_cell = tf.contrib.rnn.LSTMCell(rnn_size)
        rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=1 - dropout_prob)

        if init_state is None:
            initial_rnn_state = rnn_cell.zero_state(batch_size, dtype='float32')
        else:
            initial_rnn_state = init_state
        if traspose_dims is not None:
            in_reshaped = tf.transpose(in_reshaped, perm=traspose_dims)
        outputs, final_rnn_state = tf.nn.dynamic_rnn(rnn_cell, inputs=in_reshaped,
                                                     initial_state=initial_rnn_state, dtype=tf.float32)
        return outputs, final_rnn_state
