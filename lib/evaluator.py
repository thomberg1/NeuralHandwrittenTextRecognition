import torch
from torch.autograd import Variable

#######################################################################################################################

class Evaluator(object):
    def __init__(self, model, loader, criterion, decoder, scorer):
        self.model = model
        self.loader = loader
        self.criterion = criterion
        self.decoder = decoder
        self.scorer = scorer
        self.use_cuda = next(self.model.parameters()).is_cuda


    def evaluate(self):
        self.model.eval()
        with torch.no_grad():

            idx_batch, avg_loss, avg_score, avg_cer = 0, 0.0, 0.0, 0.0
            for idx_batch,(input_vars, target_vars, input_size, target_size) in enumerate(self.loader):
                input_vars = Variable(input_vars)
                if self.use_cuda:
                    input_vars = input_vars.cuda()

                preds = self.model(input_vars)

                # preds : seqLength x batchSize x alphabet_size
                # preds_size : batchSize
                preds = preds.transpose(1,0)
                preds_size = torch.IntTensor(preds.size(1)).fill_(preds.size(0))

                targets = self.decoder.encode(target_vars, target_size)

                loss = self.criterion(preds, targets, preds_size, target_size)
                avg_loss += (loss.item() / input_vars.size(0))

                decoder_seq, target_seq = self.decoder(preds, target_vars, preds_size, target_size)

                acc, cer = self.scorer.score(decoder_seq, target_seq)
                avg_score += acc
                avg_cer += cer

                del preds
                del loss

            avg_loss = avg_loss / (idx_batch + 1)
            avg_score = avg_score / (idx_batch + 1)
            avg_cer = avg_cer / (idx_batch + 1)

            return avg_loss, avg_score, avg_cer

#######################################################################################################################
