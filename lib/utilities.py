import matplotlib.pyplot as plt

import numpy as np

import torch
from torch import nn
import torch.nn.init as init
import torchvision.transforms as transforms

from pycuda import autoinit, driver


#######################################################################################################################

def gpu_stat():
    if torch.cuda.is_available():

        def pretty_bytes(bytes, precision=1):
            abbrevs = ((1<<50, 'PB'),(1<<40, 'TB'),(1<<30, 'GB'),(1<<20, 'MB'),(1<<10, 'kB'),(1, 'bytes'))
            if bytes == 1:
                return '1 byte'
            for factor, suffix in abbrevs:
                if bytes >= factor:
                    break
            return '%.*f%s' % (precision, bytes / factor, suffix)


        device = autoinit.device
        print( 'GPU Name: %s' % device.name())
        print( 'GPU Memory: %d' % pretty_bytes(device.total_memory()))
        print( 'CUDA Version: %s' % str(driver.get_version()))
        print( 'GPU Free/Total Memory: %d%%' % ((driver.mem_get_info()[0] /driver.mem_get_info()[1]) * 100))

#######################################################################################################################

class HYPERPARAMETERS(dict):
    """
    Class to make it easier to access hyper parameters by either dictionary or attribute syntax.
    """
    def __init__(self, dictionary):
        super(HYPERPARAMETERS, self).__init__(dictionary)
    def __getattr__(self, name):
        return self[name]
    def __setattr__(self, name, value):
        self[name] = value
    def __getstate__(self):
        return self
    def __setstate__(self, d):
        self = d

#######################################################################################################################

class Metric(object):
    """
    Class to track runtime statistics easier. Inspired by History Variables that not only store the current value,
    but also the values previously assigned. (see https://rosettacode.org/wiki/History_variables)
    """
    def __init__(self, metrics):
        self.metrics = [m[0] for m in metrics]
        self.init_vals = { m[0] : m[1] for m in metrics}
        self.values = {}
        for name in self.metrics:
            self.values[name] = []

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        if name in self.metrics:
            self.values[name].append(value)

    def __getattr__(self, attr):
        if attr in self.metrics and not len(self.values[attr]):
            val = self.init_vals[attr]
        else:
            val = self.__dict__[attr]
        return val

    def values(self, metric):
        return self.values[metric]

    def state_dict(self):
        state = {}
        for m in self.metrics:
            state[m] = self.values[m]
        return state

    def load_state_dict(self, state_dict):
        for m in state_dict:
            self.values[m] = state_dict[m]

#######################################################################################################################
# https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5

def torch_weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal(m.weight.data, mean=1, std=0.02)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal(m.weight.data, mean=1, std=0.02)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal(m.weight.data, mean=1, std=0.02)
        init.constant(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal(m.weight.data)
        init.normal(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal(param.data)
            else:
                init.normal(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal(param.data)
            else:
                init.normal(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal(param.data)
            else:
                init.normal(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal(param.data)
            else:
                init.normal(param.data)

#######################################################################################################################
# https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py

class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.contiguous().view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

#######################################################################################################################
# https://stackoverflow.com/questions/842557/how-to-prevent-a-block-of-code-from-being-interrupted-by-keyboardinterrupt-in-py

import signal

class DelayedKeyboardInterrupt(object):
    def __init__(self):
        self.signal_received = None

    def __enter__(self):
        self.signal_received = None
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        print('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)

#######################################################################################################################

def visualize_data(img, tks, vocab, figsize=None, ax=None):
    if img.size(0) is 3:
        pil_img = transforms.ToPILImage()(img)
        pil_img = pil_img.convert('L')
        img = transforms.ToTensor()(pil_img)

    img = img.squeeze().cpu().numpy()
    if isinstance(tks, str):
        txt = tks
    else:
        txt = ''.join([vocab.idx2token[tkn.item()] for tkn in tks])

    if not figsize is None:
        plt.figure(figsize=figsize)

    if not ax:
        plt.title(txt)
        plt.imshow(img, cmap='gray')
    else:
        ax.set_title(txt)
        ax.imshow(img, cmap='gray')
        
#######################################################################################################################

# https://stackoverflow.com/questions/42480111/model-summary-in-pytorch
# https://github.com/fchollet/keras/blob/master/keras/utils/layer_utils.py

def print_model_summary(model, line_length=None, positions=None, print_fn=print):
    """Prints a summary of a model.
    # Arguments
        model: model instance.
        line_length: Total length of printed lines
            (e.g. set this to adapt the display to different
            terminal window sizes).
        positions: Relative or absolute positions of log elements in each line.
            If not provided, defaults to `[.33, .55, .67, 1.]`.
        print_fn: Print function to use.
            It will be called on each line of the summary.
            You can set it to a custom function
            in order to capture the string summary.
    """

    line_length = line_length or 65
    positions = positions or [.45, .85, 1.]
    if positions[-1] <= 1:
        positions = [int(line_length * p) for p in positions]
    # header names for the different log elements
    to_display = ['Layer (type)', 'Shape', 'Param #']

    def print_row(fields, positions):
        line = ''
        for i in range(len(fields)):
            if i > 0:
                line = line[:-1] + ' '
            line += str(fields[i])
            line = line[:positions[i]]
            line += ' ' * (positions[i] - len(line))
        print_fn(line)

    print_fn( "Summary for model: " + model.__class__.__name__)
    print_fn('_' * line_length)
    print_row(to_display, positions)
    print_fn('=' * line_length)

    def print_module_summary(name, module):
        count_params = sum([np.prod(p.size()) for p in module.parameters()])
        output_shape = tuple([tuple(p.size()) for p in module.parameters()])
        cls_name = module.__class__.__name__
        fields = [name + ' (' + cls_name + ')', output_shape, count_params]
        print_row(fields, positions)

    module_count = len(set(model.modules()))
    for i, item in enumerate(model.named_modules()):
        name, module = item
        cls_name = str(module.__class__)
        if not 'torch' in cls_name or 'container' in cls_name:
            continue

        print_module_summary(name, module)
        if i == module_count - 1:
            print_fn('=' * line_length)
        else:
            print_fn('_' * line_length)

    trainable_count = 0
    non_trainable_count = 0
    for name, param in model.named_parameters():
        if 'bias' in name or 'weight' in name :
            trainable_count += np.prod(param.size())
        else:
            non_trainable_count += np.prod(param.size())

    print_fn('Total params:         {:,}'.format(trainable_count + non_trainable_count))
    print_fn('Trainable params:     {:,}'.format(trainable_count))
    print_fn('_' * line_length)

#######################################################################################################################

def plot_learning_curves(m, loss_ylim=(0, 9.0), score_ylim=(0.0, 1.0), figsize=(14,6)):
    train_loss = m.values['train_loss']
    train_score = None
    train_lr = m.values['train_lr']
    valid_loss = m.values['valid_loss']
    valid_score = m.values['valid_score']
    valid_cer = m.values['valid_cer']

    train_epochs = np.linspace(1, len(train_loss), len(train_loss))

    fig, ax = plt.subplots(1,2,figsize=figsize)

    if not train_loss is None:
        loss_train_min = np.min(train_loss)
        ax[0].plot(train_epochs, train_loss, color="r",
                   label="Trainings loss (min %.4f)" % loss_train_min) #alpha=0.3)

    if not valid_loss is None:
        loss_valid_min = np.min(valid_loss)
        ax[0].plot(train_epochs, valid_loss, color="b",
                   label="Validation loss (min %.4f)" % loss_valid_min) #alpha=0.3)
        ax[0].legend(loc="best")

    if not train_lr is None:
        ax0 = ax[0].twinx()
        ax0.plot(train_epochs, train_lr, color="g", label="Learning Rate") #alpha=0.3)
        ax0.set_ylabel('learning rate')

    ax[0].set_title("Loss")
    #     ax[0].set_xlim(0, np.max(train_epochs))
    #    ax[0].set_ylim(*loss_ylim)
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('loss')

    if not train_score is None:
        score_train_max = np.max(train_score)
        ax[1].plot(train_epochs, train_score, color="r",
                   label="Trainings score (max %.4f)" % score_train_max)

    if not valid_score is None:
        score_valid_max = np.max(valid_score)
        ax[1].plot(train_epochs, valid_score, color="b",
                   label="Validation score (max %.4f)" % score_valid_max)

    if not valid_cer is None:
        score_cer_max = np.max(valid_cer)
        ax[1].plot(train_epochs, valid_cer, color="b",
                   label="Validation cer (max %.4f)" % score_cer_max)

    if not train_lr is None:
        ax1 = ax[1].twinx()
        ax1.plot(train_epochs, train_lr, color="g", label="Learning Rate") #alpha=0.3)
        ax1.set_ylabel('learning rate')

    ax[1].set_title("Score")
    #     ax[1].set_xlim(0, np.max(train_epochs))
    #     ax[1].set_ylim(*score_ylim)
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('score')
    ax[1].legend(loc="best")

    plt.grid(False)
    plt.tight_layout()
    plt.show()

#######################################################################################################################
