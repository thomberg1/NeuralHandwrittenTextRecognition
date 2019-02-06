import time
import os
import shutil
import torch


#######################################################################################################################
# https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/util/checkpoint.py

class Checkpoint(object):
    """
    Class that manages the saving and loading of a model during training. It allows training to be suspended
    and resumed at a later time.
    """

    def __init__(self, module, optimizer, stopping, metrics, experiment_dir="model", checkpoint_file=None,
                 restore_from=-1, interval=10, verbose=0):

        self.CHECKPOINT_DIR_NAME = 'chkpt'
        self.CHECKPOINT_FILE_NAME = 'state.tar'

        self.module = module
        self.optimizer = optimizer
        self.stopping = stopping
        self.metrics = metrics
        self.interval = interval

        self.experiment_dir = experiment_dir
        self.checkpoint_file = checkpoint_file
        self.restore_from = restore_from
        self.verbose = verbose

        self.timsetamp = None

    def create(self, epoch):
        """
        Creates a checkpoint of the current model and related training parameters into a subdirectory of the checkpoint
        directory. The name of the subdirectory is the current local time in Y_M_D_H_M_S format.
        """

        self.timsetamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        path = os.path.join(self.CHECKPOINT_DIR_NAME, self.experiment_dir, self.timsetamp)

        if os.path.exists(path):
            shutil.rmtree(path)

        os.makedirs(path)

        state = {
            'timsetamp': self.timsetamp,
            'epoch': epoch,
            'module': self.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'stopping': self.stopping.state_dict(),
            'metrics': self.metrics.state_dict()
        }

        torch.save(state, os.path.join(path, self.CHECKPOINT_FILE_NAME))

        if self.verbose:
            print("Created checkpoint in '{}' ".format(path))

    def restore(self):
        """
        Restores a current model and related training parameters from a checkpoint object that was previously
        saved to disk.
        """

        file_name = self.last()

        assert not file_name is None

        state = torch.load(file_name)

        self.timsetamp = state['timsetamp']
        self.module.load_state_dict(state['module'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.stopping.load_state_dict(state['stopping'])
        self.metrics.load_state_dict(state['metrics'])

        if self.verbose:
            print("Restored checkpoint from '{}' ".format(file_name))

        return state['epoch']

    def step(self, epoch):
        """"""
        if not epoch % self.interval:
            self.create(epoch)
            if self.verbose:
                print("Epoch: %d checkpoint created!" % epoch)

    def last(self):
        """
        Returns the path to the last saved checkpoint file for a given set of parameters.
        Precondition: at least one checkpoint has been made (i.e., latest checkpoint subdirectory exists).
         """
        checkpoints_path = os.path.join(self.CHECKPOINT_DIR_NAME, self.experiment_dir)

        try:
            path = sorted(os.listdir(checkpoints_path), reverse=False)[self.restore_from]

            last_path = os.path.join(checkpoints_path, os.path.join(path, self.CHECKPOINT_FILE_NAME))
        except:
            last_path = "undefined"

        return last_path

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Timestamp: {}\n'.format(self.timsetamp)
        fmt_str += '    Last Checkpoint: {}\n'.format(self.last())
        return fmt_str

#######################################################################################################################
