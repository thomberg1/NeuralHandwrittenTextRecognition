import math
import numpy as np
import time

import torch
from torch import optim
from torch.autograd import Variable
import torchvision.transforms as transforms

from warpctc_pytorch import CTCLoss

from utilities import HYPERPARAMETERS, Metric, torch_weight_init, print_model_summary, DelayedKeyboardInterrupt, \
    plot_learning_curves
from .vocabulary import Vocabulary
from .dataloader import DataArgumentation, FromNumpyToTensor, IAMHandwritingDataset, alignCollate
from .model_crnn import CRNN
from .checkpoint import Checkpoint
from .decoder import CTCGreedyDecoder
from .stopping import Stopping
from .logger import PytorchLogger
from .spelling import Spelling

#######################################################################################################################

# https://github.com/SeanNaren/deepspeech.pytorch/blob/master/decoder.py

import Levenshtein as lev


def score(predicted_vars, target_vars):
    idx, acc, cer = 0, 0.0, 0.0
    for idx, (dseq, tseq) in enumerate(zip(predicted_vars, target_vars)):
        acc += 1 if dseq == tseq else 0
        cer += lev.distance(dseq, tseq) / float(len(tseq))

    return acc / (idx + 1), cer / (idx + 1)


def score_spell(predicted_vars, target_vars, sp):
    idx, acc, acc2, cer = 0, 0.0, 0.0, 0.0
    for idx, (dseq, tseq) in enumerate(zip(predicted_vars, target_vars)):
        acc += 1 if dseq == tseq else 0
        if dseq in sp.words:
            acc2 += 1 if dseq == tseq else 0
        else:
            acc2 += 1 if sp.correction(dseq) == tseq else 0

        cer += lev.distance(dseq, tseq) / float(len(tseq))

    return acc / (idx + 1), acc2 / (idx + 1), cer / (idx + 1)


#######################################################################################################################

# torch.cuda.is_available = lambda : False
# torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True

seed = 0
np.random.seed(0)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

H = HYPERPARAMETERS({
    'MODEL_PATH': './chkpt/best_crnn_model.tar',
    'ROOT_DIR': './data',
    'BATCH_SIZE': 16,
    'HEIGHT': 64,
    'NUM_WORKERS': 8,
    'HIDDEN_SIZE': 256,
    'NUM_LAYERS': 2,
    'RNN_DROPOUT': 0.5,
    'LR': 0.0003,
    'LR_LAMBDA': lambda epoch: max(math.pow(0.78, math.floor((1 + epoch) / 4.0)), 0.1),
    'WEIGHT_DECAY': 0,
    'MAX_GRAD_NORM': 5.,
    'ARGUMENTATION': 0.7,
    'STOPPING_PATIENCE': 10,
    'NUM_EPOCHS': 3,
    'CHECKPOINT_FILE': 'IAM_Handwriting_Recognition_CRNN_Final_Version',
    'CHECKPOINT_INTERVAL': 5,
    'CHECKPOINT_RESTORE': False,
    'USE_CUDA': torch.cuda.is_available(),
})

print(H)

#######################################################################################################################

vocab = Vocabulary(H.ROOT_DIR)
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

train_dataset = IAMHandwritingDataset(H.ROOT_DIR, vocab, dataset="train",
                                      transform=image_transform_train, target_transform=target_transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=H.BATCH_SIZE, num_workers=H.NUM_WORKERS, shuffle=True,
    collate_fn=alignCollate(img_height=H.HEIGHT), pin_memory=True)

print(train_dataset)
print(len(train_loader))

valid_dataset = IAMHandwritingDataset(H.ROOT_DIR, vocab, dataset="valid",
                                      transform=image_transform_test, target_transform=target_transform)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=H.BATCH_SIZE, shuffle=False, num_workers=H.NUM_WORKERS,
    collate_fn=alignCollate(img_height=H.HEIGHT), pin_memory=True)

print(valid_dataset)
print(len(valid_loader))

test_dataset = IAMHandwritingDataset(H.ROOT_DIR, vocab, dataset="test",
                                     transform=image_transform_test, target_transform=target_transform)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=H.BATCH_SIZE, shuffle=False, num_workers=H.NUM_WORKERS,
    collate_fn=alignCollate(img_height=H.HEIGHT))

print(test_dataset)
print(len(test_loader))

#######################################################################################################################


m = Metric([('train_loss', np.inf), ('train_score', np.inf), ('valid_loss', np.inf), ('valid_score', 0),
            ('train_lr', 0), ('valid_cer', np.inf)])

crnn = CRNN(num_classes=len(vocab), hidden_size=H.HIDDEN_SIZE, num_layers=H.NUM_LAYERS, rnn_dropout=H.RNN_DROPOUT,
            initialize=torch_weight_init)
if H.USE_CUDA:
    crnn.cuda()

print_model_summary(crnn)
print(crnn)

optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, crnn.parameters())),
                       amsgrad=False,
                       betas=(0.9, 0.999),
                       eps=1e-08,
                       lr=H.LR,
                       weight_decay=H.WEIGHT_DECAY)

criterion = CTCLoss()

stopping = Stopping(crnn, patience=H.STOPPING_PATIENCE)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[H.LR_LAMBDA])

decoder = CTCGreedyDecoder(vocab)

logger = PytorchLogger()

checkpoint = Checkpoint(crnn, optimizer, stopping, m, experiment_dir="final", checkpoint_file=H.CHECKPOINT_FILE,
                        restore_from=-1, interval=H.CHECKPOINT_INTERVAL, verbose=0)

#######################################################################################################################

epoch_start = checkpoint.restore() if H.CHECKPOINT_RESTORE else 1

epoch_itr = logger.set_itr(range(epoch_start, H.NUM_EPOCHS + 1))

epoch = 0
for epoch in epoch_itr:

    with DelayedKeyboardInterrupt():

        crnn.train()

        scheduler.step(epoch)

        m.train_lr = [float(param_group['lr']) for param_group in optimizer.param_groups][0]

        idx_batch, avg_loss = 0, 0.0
        for idx_batch, (input_vars, target_vars, input_size, target_size) in enumerate(train_loader):
            input_vars = Variable(input_vars)
            if H.USE_CUDA:
                input_vars = input_vars.cuda()

            preds = crnn(input_vars)

            preds = preds.transpose(1, 0)
            preds_size = torch.IntTensor(preds.size(1)).fill_(preds.size(0))

            targets = decoder.encode(target_vars, target_size)

            loss = criterion(preds, targets, preds_size, target_size)
            avg_loss += (loss.item() / input_vars.size(0))

            crnn.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(crnn.parameters(), H.MAX_GRAD_NORM)
            optimizer.step()

            del preds
            del loss

        m.train_loss = avg_loss / (idx_batch + 1)

        #  -----  validation

        crnn.eval()
        with torch.no_grad():

            idx_batch, avg_loss, avg_score, avg_cer = 0, 0.0, 0.0, 0.0
            for idx_batch, (input_vars, target_vars, input_size, target_size) in enumerate(valid_loader):
                input_vars = Variable(input_vars)
                if H.USE_CUDA:
                    input_vars = input_vars.cuda()

                preds = crnn(input_vars)

                preds = preds.transpose(1, 0)
                preds_size = torch.IntTensor(preds.size(1)).fill_(preds.size(0))

                targets = decoder.encode(target_vars, target_size)

                loss = criterion(preds, targets, preds_size, target_size)
                avg_loss += (loss.item() / input_vars.size(0))

                decoder_vars = preds.max(2)[1].transpose(1, 0)
                decoder_seq = decoder.decode(decoder_vars.cpu(), remove_repetitions=True, filler='')
                target_seq = decoder.decode(target_vars.cpu(), remove_repetitions=False, filler='')

                acc, cer = score(decoder_seq, target_seq)
                avg_score += acc
                avg_cer += cer

                del preds
                del loss

            m.valid_loss = avg_loss / (idx_batch + 1)

            m.valid_score = avg_score / (idx_batch + 1)
            m.valid_cer = avg_cer / (idx_batch + 1)

        epoch_itr.log_values(m.train_loss, m.train_score, m.train_lr, m.valid_loss, m.valid_score,
                             stopping.best_score_epoch, stopping.best_score)

        if checkpoint:
            checkpoint.step(epoch)

        if stopping.step(epoch, 0, m.valid_score):
            print("Early stopping at epoch: %d, score %f" % (stopping.best_score_epoch, stopping.best_score))
            break

checkpoint.create(epoch)

time.sleep(2)  # wait for tqm to settle

#######################################################################################################################

print(logger)
print(stopping)
print(checkpoint)
plot_learning_curves(m)

#######################################################################################################################

crnn.load_state_dict(stopping.best_score_state)

torch.save(crnn.state_dict(), H.MODEL_PATH)

#######################################################################################################################
words = [a['transcript'] for a in train_dataset.annotations]
print(len(words))

sp = Spelling("./data", words)
print(sp)

crnn_pred = CRNN(num_classes=len(vocab), hidden_size=H.HIDDEN_SIZE, num_layers=H.NUM_LAYERS, rnn_dropout=H.RNN_DROPOUT,
                 initialize=torch_weight_init)
if H.USE_CUDA:
    crnn_pred.cuda()

state = torch.load(H.MODEL_PATH)
crnn_pred.load_state_dict(state)

crnn.eval()
with torch.no_grad():
    idx_batch = 0
    avg_score, avg_score2, avg_cer = 0.0, 0.0, 0.0
    for idx_batch, (input_vars, target_vars, input_size, target_size) in enumerate(test_loader):
        input_vars = Variable(input_vars)
        if H.USE_CUDA:
            input_vars = input_vars.cuda()

        preds = crnn(input_vars)
        preds = preds.transpose(1, 0)

        decoder_vars = preds.max(2)[1].transpose(1, 0)
        decoder_seq = decoder.decode(decoder_vars.cpu(), remove_repetitions=True, filler='')
        target_seq = decoder.decode(target_vars.cpu(), remove_repetitions=False, filler='')

        acc, acc2, cer = score_spell(decoder_seq, target_seq, sp)
        avg_score += acc
        avg_score2 += acc2
        avg_cer += cer

print('Test Results:')
print('     Accuracy:       {:.4f}'.format(avg_score / (idx_batch + 1)))
print('     Accuracy/spell: {:.4f}'.format(avg_score2 / (idx_batch + 1)))
print('     CER:            {:.4f}'.format((avg_cer / (idx_batch + 1))))
