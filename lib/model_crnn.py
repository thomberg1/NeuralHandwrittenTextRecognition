import math

import torch
from torch import nn
import torchvision.transforms as transforms

from .utilities import SequenceWise, torch_weight_init, HYPERPARAMETERS, print_model_summary
from .vocabulary import Vocabulary
from .dataloader import DataArgumentation, FromNumpyToTensor, IAMHandwritingDataset, ResizeAndPad, alignCollate

#######################################################################################################################

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class CNN(nn.Module):
    def __init__(self, initialize=None):
        super(CNN, self).__init__()
        self.initialize = initialize
        self.inplanes = 32

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32, affine=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.bb1 = self._make_layer(BasicBlock, 64, 2)
        self.maxpoo11 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.dropout1 = nn.Dropout(0.5)

        self.bb2 = self._make_layer(BasicBlock, 128, 2)
        self.maxpoo12 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.dropout2 = nn.Dropout(0.5)

        self.bb3 = self._make_layer(BasicBlock, 256, 2)
        self.maxpoo13 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.dropout3 = nn.Dropout(0.5)

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 0)),
            nn.Dropout(0.5)
        )

        if not self.initialize is None:
            self.initialize(self)

    def forward(self, input_vars):
        conv = self.conv1(input_vars)
        conv = self.bb1(conv)
        conv = self.maxpoo11(conv)
        conv = self.dropout1(conv)
        conv = self.bb2(conv)
        conv = self.maxpoo12(conv)
        conv = self.dropout2(conv)
        conv = self.bb3(conv)
        conv = self.maxpoo13(conv)
        conv = self.dropout3(conv)
        conv = self.conv2(conv)
        return conv.squeeze(2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

#######################################################################################################################

class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, dropout=0.0, initialize=None):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.initialize = initialize

        self.rnn = nn.GRU(self.input_size, self.hidden_size, num_layers=self.num_layers, batch_first=True,
                          bidirectional=True, dropout=dropout)

        fully_connected = nn.Linear(self.hidden_size * 2, self.num_classes)
        self.fc = SequenceWise(fully_connected)

        if not self.initialize is None:
            self.initialize(self)

    def forward(self, input_vars):

        recurrent, _ = self.rnn(input_vars)

        output = self.fc(recurrent)

        return output

#######################################################################################################################

class CRNN(nn.Module):

    def __init__(self, num_classes, hidden_size, num_layers=1, rnn_dropout=0.0, initialize=None):
        super(CRNN, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_dropout = rnn_dropout

        self.cnn = CNN(initialize=initialize)

        self.rnn = BatchRNN(256, self.hidden_size, self.num_classes, num_layers=self.num_layers,
                            dropout=self.rnn_dropout, initialize=initialize)

    def forward(self, input_vars):

        conv = self.cnn(input_vars)

        conv = conv.permute(0, 2, 1)  # [b, w, c]

        output = self.rnn(conv)

        return output

#######################################################################################################################
if __name__ == '__main__':

    H = HYPERPARAMETERS({
        'MODEL_PATH': './chkpt/IAM_Handwriting_Recognition_CRNN_new_model2.tar',
        'ROOT_DIR': './data',
        'EXPERIMENT': 'new_model2',
        'BATCH_SIZE': 16,
        'HEIGHT': 64,
        'PADDING': 10,
        'NUM_WORKERS': 8,
        'HIDDEN_SIZE': 256,
        'NUM_LAYERS': 2,
        'RNN_DROPOUT': 0.5,
        'LR': 0.0003,
        'LR_LAMBDA': lambda epoch: max(math.pow(0.78, math.floor((1 + epoch) / 11.0)), 0.4),
        'WEIGHT_DECAY': 0,
        'MAX_GRAD_NORM': 5.,
        'ARGUMENTATION': 0.9,
        'STOPPING_PATIENCE': 10,
        'NUM_EPOCHS': 50,

        'CHECKPOINT_FILE': 'IAM_Handwriting_Recognition_CRNN_new_model2',
        'CHECKPOINT_INTERVAL': 5,
        'CHECKPOINT_RESTORE': False,

        'USE_CUDA': torch.cuda.is_available(),
    })

    vocab = Vocabulary("./data")
    vocab.load()
    print(vocab)


    image_transform_train = transforms.Compose([
        transforms.Pad(2, fill=255),
        DataArgumentation(threshold=H.ARGUMENTATION),
        transforms.ToTensor(),
    ])

    image_transform_test = transforms.Compose([
        transforms.Pad(2, fill=255),
        transforms.ToTensor(),
    ])

    target_transform = transforms.Compose([
        FromNumpyToTensor()
    ])

    train_dataset = IAMHandwritingDataset('./data', vocab, dataset="train",
                                          transform=image_transform_train, target_transform=target_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=H.BATCH_SIZE, num_workers=H.NUM_WORKERS, shuffle=True,
        collate_fn=alignCollate(img_height=H.HEIGHT), pin_memory=True)

    print(train_dataset)
    print(len(train_loader))


    ####################################################################################################################

    cnn_cpu = CNN(initialize=torch_weight_init)

    input_vars_cpu, _, _, _ = next(train_loader.__iter__())

    cnn_outputs_cpu = cnn_cpu(input_vars_cpu[0:2])

    print_model_summary(cnn_cpu)
    print(cnn_cpu)

    print(input_vars_cpu[0:2].shape)
    print(cnn_outputs_cpu.shape)

    ####################################################################################################################

    crnn_cpu = CRNN(num_classes=56, hidden_size=256, num_layers=2, rnn_dropout=0.5, initialize=torch_weight_init)

    input_vars_cpu, _, _, _ = next(train_loader.__iter__())

    cnn_outputs_cpu = crnn_cpu(input_vars_cpu[0:2])

    print_model_summary(crnn_cpu)
    print(crnn_cpu)

    print(input_vars_cpu[0:2].shape)
    print(cnn_outputs_cpu.shape)