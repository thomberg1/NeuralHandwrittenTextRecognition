import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import random

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

class Encoder(nn.Module):
    def __init__(self, hidden_size=128, n_layers=1, dropout=0, initialize=None,
                 bidirectional=True):
        super(Encoder,self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.initialize = initialize
        self.bidirectional = bidirectional
        self.input_size = 256

        self.cnn = CNN(initialize=self.initialize)

        self.rnn = nn.GRU(self.input_size, self.hidden_size, self.n_layers, batch_first=True, dropout=self.dropout,
                          bidirectional=self.bidirectional)

        if not self.initialize is None:
            self.initialize(self)

    def forward(self, input_vars, hidden):

        internal_vars = self.cnn(input_vars)

        internal_vars = internal_vars.transpose(1,2)

        output_vars, hidden = self.rnn(internal_vars, hidden)

        output_vars = self._sum_outputs(output_vars)

        hidden = self._cat_hidden(hidden)

        return output_vars, hidden


    def _cat_hidden(self, h):
        """
        (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _sum_outputs(self, outputs):
        # sum bidirectional outputs
        if self.bidirectional:
            outputs = (outputs[:, :, :self.hidden_size] +
                       outputs[:, :, self.hidden_size:])
        return outputs

    def initHidden(self,batch_size,use_cuda=False):
        m = 2 if self.bidirectional else 1
        h0 = Variable(torch.zeros(self.n_layers * m, batch_size, self.hidden_size))
        if use_cuda:
            return h0.cuda()
        else:
            return h0


class Attention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, initialize=None):
        super(Attention, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.initialize = initialize
        self.mode = 'non_linear' # 'dot' , 'linear', 'non_linear'


        if self.mode == 'dot':
            pass
        elif self.mode == 'linear':
            self.fc = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size,
                                bias=False)
        elif self.mode == 'non_linear':
            self.fc1 = nn.Linear(self.decoder_hidden_size, self.encoder_hidden_size,bias=False)
            self.fc2 = nn.Linear(self.encoder_hidden_size, 1, bias=False)
        else:
            raise NotImplementedError

        self.linear_out = nn.Linear(self.decoder_hidden_size, self.encoder_hidden_size)
        self.mask = None

        if not self.initialize is None:
            self.initialize(self)

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        input_size = context.size(1)

        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)

        if self.mode == 'dot':
            attn = torch.bmm(output, context.transpose(1, 2))
        elif self.mode == 'linear':
            output = self.fc(output)
            attn = torch.bmm(output, context.transpose(1, 2))
        elif self.mode == 'non_linear':
            #             print("non_linear1", context.squeeze(0).shape, output.squeeze(0).shape)
            out_expanded = output.squeeze(0).expand_as(context.squeeze(0))
            #             print("non_linear2", context.shape, out_expanded.shape)
            energy = torch.cat((context, out_expanded), 2)
            #             print("non_linear3", energy.shape)
            energy = torch.tanh(self.fc1(energy))
            #             print("non_linear4", energy.shape)
            attn = self.fc2(energy).transpose(1, 2)
        #             print("non_linear5", attn.shape)

        else:
            raise NotImplementedError

        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = self.linear_out(combined.view(-1, self.decoder_hidden_size))
        output = torch.tanh(output).view(batch_size, -1, self.encoder_hidden_size)

        return output


class Decoder(nn.Module):
    def __init__(self, input_vocab_size, decoder_hidden_size, encoder_hidden_size,
                 output_size, n_layers=1, dropout=0.0, initialize=None):
        super(Decoder, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.initialize = initialize

        self.embedding = nn.Sequential(
            nn.Embedding(self.input_vocab_size, self.decoder_hidden_size),
            nn.Dropout(0.5)
        )

        self.attn = Attention(self.encoder_hidden_size, self.decoder_hidden_size, self.initialize)

        self.rnn = nn.GRU(self.decoder_hidden_size, self.decoder_hidden_size,
                          self.n_layers, batch_first=True, dropout=self.dropout)

        self.fc = nn.Linear(self.encoder_hidden_size, self.output_size)

        if not self.initialize is None:
            self.initialize(self)

        fix_embedding = torch.from_numpy(np.eye(self.input_vocab_size, self.input_vocab_size).astype(np.float32))
        self.embedding.weight = nn.Parameter(fix_embedding)
        self.embedding.weight.requires_grad=False

    def forward(self, input_vars, last_hidden, encoder_outputs):

        batch_size = input_vars.size(0)
        output_size = input_vars.size(1)

        embedding_input = self.embedding(input_vars)

        output, hidden = self.rnn(embedding_input, last_hidden)

        output = output[:, :, :self.encoder_hidden_size] + output[:, :, self.encoder_hidden_size :]

        output = self.attn(output, encoder_outputs)

        output = self.fc(output.contiguous().view(-1, self.encoder_hidden_size))

        output = F.log_softmax(output, dim=1).view(batch_size, output_size, -1)

        return output, hidden

class NeuralHandwrittenTextRecognizer(nn.Module):
    def __init__(self, encoder, decoder, vocab, max_seq_len, teacher_forcing_ratio=0.5):
        super(NeuralHandwrittenTextRecognizer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, input_var, target_var):
        use_cuda = next(self.parameters()).is_cuda
        batch_size = input_var.size(0)
        max_length = target_var.size(1) if not target_var is None else self.max_seq_len + 2

        init_hidden = self.encoder.initHidden(batch_size, use_cuda=use_cuda)
        encoder_outputs, encoder_hidden = self.encoder(input_var, init_hidden)

        decoder_input = Variable(torch.LongTensor([self.vocab("<SOS>")] * batch_size)).view(batch_size, 1)
        if use_cuda:
            decoder_input = decoder_input.cuda()

        decoder_hidden = encoder_hidden

        decoder_outputs = []
        for di in range(max_length-1):

            decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                          decoder_hidden,
                                                          encoder_outputs)

            step_output = decoder_output.squeeze(1)
            decoder_outputs.append(step_output)

            use_teacher_forcing = True if self.training and random.random() < self.teacher_forcing_ratio else False
            if use_teacher_forcing:
                decoder_input = Variable(target_var.data[:, di+1]).view(batch_size, 1)
            else:
                decoder_input = Variable(decoder_outputs[-1].topk(1)[1].data).view(batch_size, 1)
            if next(self.parameters()).is_cuda:
                decoder_input = decoder_input.cuda()

        return decoder_outputs
