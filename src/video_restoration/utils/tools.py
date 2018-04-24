import copy
import logging
import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def variable_summaries(var, name):
    """

    Method for saving tensorboard summaries. During training can be used inside layer method,
    Summaries like mean, stddev, max, min value will be printed in tensorboard graph.

    :param var: Tensorflow tensor
    :param name: Summary name
    :return:
    """

    with tf.name_scope('summaries_{}'.format(name)):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram(name, var)


class SequentialBatchCollector(object):

    def __init__(self, iterator, batch_num=128):
        self.iterator = iterator
        self.batch_num = batch_num

    def collect_batch(self):
        # first_elem = next(self.iterator.iter_items(), None)
        first_elem = next(self.iterator.iter_items())
        elem_shape_x = first_elem.shape
        elem_shape_y = list(copy.deepcopy(elem_shape_x))
        elem_shape_y[-1] += 1
        batch_x = np.zeros((self.batch_num, *elem_shape_x))
        batch_y = np.zeros((self.batch_num, *elem_shape_y))
        for b_idx in range(0, self.batch_num):
            batch_x[b_idx] = next(self.iterator.iter_items())
            batch_y[b_idx] = np.concatenate([batch_x[b_idx], np.zeros((*elem_shape_x[:-1], 1))], axis=-1)
        return batch_x, batch_y


class Messenger(object):

    def __init__(self):
        pass

    @staticmethod
    def set_logger_path(path):
        handler = logging.FileHandler(path)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    @staticmethod
    def text(message):
        logger.info(message)

    @staticmethod
    def section_message(message):
        logger.info(message)
        logger.info("-" * 90)

    @staticmethod
    def fancy_message(message):
        number_of_letters = len(message)
        side_ms = int((90 - number_of_letters) / 2.0)
        logger.info("=" * 90)
        logger.info(side_ms * '=' + " " + message + " " + side_ms * '=')
        logger.info("=" * 90)

    @staticmethod
    def title_message(message):
        logger.info("=" * 90)
        logger.info(message)
        logger.info("=" * 90)
