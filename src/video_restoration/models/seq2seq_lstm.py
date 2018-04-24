import os
import tensorflow as tf

from video_restoration.iterators.utils import InfiniteIterator
from video_restoration.iterators.luxoft_cars import LuxoftCars
from video_restoration.utils.tools import SequentialBatchCollector, Messenger

from video_restoration.layers.rnn import lstm
from video_restoration.layers.dense_layers import faltten, fully_connected_layer
from video_restoration.layers.cnn_layers import convolutional_layer
from video_restoration.layers.pooling_layers import maxpool_layer


class Seq2SeqLSTM(object):

    def __init__(self, iterator, batch_size=128, filters_num=32):
        self.iterator = iterator
        self.batch_size = batch_size

        # Define batch collector
        self.batch_collect = SequentialBatchCollector(self.iterator, batch_num=batch_size)

        # Define placeholders
        self.filters_num = filters_num
        self.input_enc = None
        self.input_dec = None
        self.saver = None
        self.encoder_state = None
        self.sess = tf.Session()

        f_batch = self.batch_collect.collect_batch()
        self.x_labels_shape = f_batch[0].shape[1:]
        self.y_labels_shape = f_batch[1].shape[1:]
        self.y_labels = tf.placeholder(tf.float32, [None, *self.y_labels_shape])

    def encoder(self, input_seq):
        reuse_conv = False
        fr_num = input_seq.get_shape().as_list()[-1]
        conv_fr_output = []
        for l_idx in range(0, fr_num):
            input_fr = input_seq[..., l_idx]
            conv1 = convolutional_layer(input_fr, [3, 3, self.filters_num], stride=1, activation="relu",
                                        name="conv1", reuse=reuse_conv)
            max_pool1 = maxpool_layer(conv1, [2, 2], stride=2, name="max_pool1")

            conv2 = convolutional_layer(max_pool1, [3, 3, self.filters_num], stride=1, activation="relu",
                                        name="conv2", reuse=reuse_conv)
            max_pool2 = maxpool_layer(conv2, [2, 2], stride=2, name="max_pool2")

            conv3 = convolutional_layer(max_pool2, [3, 3, self.filters_num], stride=1, activation="relu",
                                        name="conv3", reuse=reuse_conv)
            max_pool3 = maxpool_layer(conv3, [2, 2], stride=2, name="max_pool3")

            conv4 = convolutional_layer(max_pool3, [3, 3, self.filters_num], stride=1, activation="relu", name="conv4")
            max_pool4 = maxpool_layer(conv4, [2, 2], stride=2, name="max_pool4")

            fc1 = fully_connected_layer(faltten(max_pool4), output_size=512, activation='relu')
            reuse_conv = True
            conv_fr_output.append(fc1)
        lstm_conv_in = tf.stack(conv_fr_output, -1)

        # Encoder decoder
        lstm1, lstm1_state = lstm(lstm_conv_in, 256, 0.2, self.batch_size, name="lstm1")
        return lstm1_state

    def decoder(self, input_dec, encoder_state):
        lstm2, lstm2_state = lstm(input_dec, 256, 0.0, self.batch_size, init_state=encoder_state,
                                  traspose_dims=[0, 2, 1], name="lstm2")
        lstm3, lstm3_state = lstm(lstm2, 256, 0.0, self.batch_size, init_state=lstm2_state,
                                  preprocess_data=False, name="lstm3")
        logits = tf.layers.dense(lstm3, units=self.alphabet_size, use_bias=True)
        return logits

    def build_model(self):
        input_enc = tf.placeholder('float32', shape=[None, *self.x_labels_shape],
                                   name='input_enc')
        input_dec = tf.placeholder('float32', shape=[None, *self.x_labels_shape],
                                   name='input_dec')
        self.input_enc = input_enc
        self.input_dec = input_dec

        self.encoder_state = self.encoder(self.input_enc)
        logits = self.decoder(input_dec, self.encoder_state)
        return logits

    def restore_model(self, path):
        self.saver = tf.train.import_meta_graph(os.path.join(path, 'my-model.meta'))
        self.saver.restore(self.sess, tf.train.latest_checkpoint(path))
        graph = tf.get_default_graph()
        gvars = graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for g_v in gvars:
            Messenger.text("Variable {} restored".format(g_v))

    @staticmethod
    def cross_entropy(labels, predicted):
        # Define cross entropy
        with tf.name_scope("Cross_entropy"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicted, labels=labels)
            cross_entropy = tf.reduce_mean(cross_entropy)
            tf.summary.scalar("cross_entropy", cross_entropy)
        return cross_entropy

    @staticmethod
    def accuracy(labels, predicted):
        with tf.name_scope("Accuracy"):
            is_correct = tf.equal(tf.argmax(labels, 1), tf.argmax(predicted, 1))
            accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
            tf.summary.scalar("accuracy", accuracy)
        return accuracy

    def train(self, learning_rate=0.0001, iter_steps=20000, clip_grad=25.0, buffor_len=20,
              log_path="."):
        try:
            self.train_model(learning_rate, iter_steps, clip_grad, buffor_len, log_path)
        except KeyboardInterrupt:
            Messenger.text("Training stopped. Model saved to {}".format(os.path.join(log_path, "my-model.meta")))

    def train_model(self, learning_rate=0.0001, iter_steps=20000, clip_grad=25.0, buffor_len=20, log_path="."):
        # Build model
        y_predicted = self.build_model()

        # Save model
        self.saver = tf.train.Saver()

        # Add cross entropy to computational graph
        y_lab = tf.transpose(self.y_labels, perm=[0, 2, 1, 3])

        pre_shape = [x for x in self.y_labels.get_shape().as_list() if x != 1]
        input_shape = [-1, *pre_shape[1:]]
        y_lab = tf.reshape(y_lab, input_shape)

        y_ex_idxs = tf.argmax(y_lab, 1)
        loss = tf.contrib.seq2seq.sequence_loss(y_predicted, y_ex_idxs, tf.ones([self.batch_size,
                                                                                 self.max_seq_len]))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Clip gradient
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients = [
            None if gradient is None else tf.clip_by_norm(gradient, clip_grad)
            for gradient in gradients]
        train_step = optimizer.apply_gradients(zip(gradients, variables))

        # Initialize global variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_path)
        writer.add_graph(self.sess.graph)

        # Train loop
        cross_entropy_list = []
        for i_idx in range(iter_steps):
            # Load batch
            batch_x, batch_y = self.batch_collect.collect_batch()
            batch_x = batch_x[..., np.newaxis]
            batch_y = batch_y[..., np.newaxis]
            train_data = {self.input_enc: batch_x, self.y_labels: batch_y[:, :, 1:],
                          self.input_dec: batch_y[:, :, :-1]}

            # Train
            self.sess.run(train_step, feed_dict=train_data)
            c_train = self.sess.run(loss, feed_dict=train_data)
            cross_entropy_list = cross_entropy_list[-buffor_len:] + [c_train]
            Messenger.text("Train data Cross Entropy: {}".format(np.mean(cross_entropy_list)))

            # Save summary
            if i_idx % 25 == 0:
                self.saver.save(self.sess, os.path.join(class_log_path, 'my-model'))
                sum_res = self.sess.run(merged_summary, train_data)
                writer.add_summary(sum_res, i_idx)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from video_restoration.utils.tools import SequentialBatchCollector

    lc = LuxoftCars(data_path="/home/filip141/Datasets/Cars_Luxoft-Images", skip=2, stack=10)
    ii = InfiniteIterator(lc)
    ss = Seq2SeqLSTM(iterator=ii)
    ss.train()