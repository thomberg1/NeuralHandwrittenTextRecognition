import time
import os
import shutil

from tqdm import tqdm_notebook, tqdm

import inspect

import tensorflow as tf

#######################################################################################################################

class TensorboardLogger(object):
    def __init__(self, experiment_dir="model", verbose=0):

        self.LOG_DIR_NAME = 'logs'

        self.iterable = None
        self.experiment_dir = experiment_dir
        self.verbose = verbose

        self.last_logged_values = []
        self.epoch = -1

        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        path = os.path.join(self.LOG_DIR_NAME, self.experiment_dir, date_time)

        if os.path.exists(path):
            shutil.rmtree(path)

        os.makedirs(path)

        self.writer = tf.summary.FileWriter(path)

    def set_itr(self, iterable):
        self.iterable = iterable
        return self

    def __iter__(self):

        assert not self.iterable is None

        for obj in self.iterable:
            self.epoch = obj
            yield obj

    def log_values(self, train_loss , train_score, train_lr,
                   valid_loss, valid_score, best_score_epoch, best_score):

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        self.last_logged_values = [(i, values[i]) for i in args[1:]]

        summary = tf.Summary(value=[
            tf.Summary.Value(tag='train_loss', simple_value=train_loss),
            tf.Summary.Value(tag='train_score', simple_value=train_score),
            tf.Summary.Value(tag='train_lr', simple_value=train_lr),
            tf.Summary.Value(tag='valid_loss', simple_value=valid_loss),
            tf.Summary.Value(tag='valid_score', simple_value=valid_score),
            tf.Summary.Value(tag='best_score_epoch', simple_value=best_score_epoch),
            tf.Summary.Value(tag='best_score', simple_value=best_score)
        ])
        self.writer.add_summary(summary, self.epoch)

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Last Epoch/LR:    {} / {}\n'.format(self.epoch, self.last_logged_values[2][1])
        fmt_str += '    Train Loss/Score: {} / {}\n'.format(self.last_logged_values[0][1], self.last_logged_values[1][1])
        fmt_str += '    Valid Loss/Score: {} / {}\n'.format(self.last_logged_values[3][1], self.last_logged_values[4][1])
        fmt_str += '    Best Epoch/Score: {} / {}\n'.format(self.last_logged_values[5][1], self.last_logged_values[6][1])
        return fmt_str

#######################################################################################################################

class PytorchLogger(object):
    def __init__(self, tqdm_cls=tqdm_notebook):
        self.iterable = None
        self.last_logged_values = []
        self.epoch = -1
        self.tqdm_cls = tqdm_cls

    def set_itr(self, iterable):
        self.iterable = iterable
        self.iterable = self.tqdm_cls(iterable)
        self.iterable.set_description('Epoch')
        return self

    def __iter__(self):

        assert not self.iterable is None

        for obj in self.iterable:
            self.epoch = obj
            yield obj

    def log_values(self, train_loss , train_score, train_lr,
                   valid_loss, valid_score, best_score_epoch, best_score):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        self.last_logged_values = [(i, values[i]) for i in args[1:]]

        self.iterable.set_postfix( last="%i" % self.epoch + "/%.4f" % train_loss + "/%.4f" % train_score,
                                   lr=train_lr,
                                   best="%i" % best_score_epoch + "/%.4f" % best_score)

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Last Epoch/LR:    {} / {}\n'.format(self.epoch, self.last_logged_values[2][1])
        fmt_str += '    Train Loss/Score: {} / {}\n'.format(self.last_logged_values[0][1], self.last_logged_values[1][1])
        fmt_str += '    Valid Loss/Score: {} / {}\n'.format(self.last_logged_values[3][1], self.last_logged_values[4][1])
        fmt_str += '    Best Epoch/Score: {} / {}\n'.format(self.last_logged_values[5][1], self.last_logged_values[6][1])
        return fmt_str

#######################################################################################################################