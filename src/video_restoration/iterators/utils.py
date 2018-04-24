import numpy as np


class InfiniteIterator(object):

    def __init__(self, iterator):
        self.iterator = iterator

    def iter_items(self):
        while True:
            for x_item in self.iterator:
                yield x_item

    def __iter__(self):
        return self.iter_items()