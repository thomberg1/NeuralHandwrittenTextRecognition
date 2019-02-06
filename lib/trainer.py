import torch
from torch.autograd import Variable

#######################################################################################################################

class Trainer(object):
    def __init__(self, model, loader, optimizer, scheduler, criterion, decoder):
        self.model = model
        self.loader = loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.decoder = decoder
        self.use_cuda = next(self.model.parameters()).is_cuda

    def train(self, epoch, max_grade_norm):
        self.model.train()

        self.scheduler.step(epoch)

        train_lr = [float(param_group['lr']) for param_group in self.optimizer.param_groups][0]

        idx_batch, avg_loss = 0, 0.0
        for idx_batch,(input_vars, target_vars, input_size, target_size) in enumerate(self.loader):
            input_vars = Variable(input_vars)
            if self.use_cuda:
                input_vars = input_vars.cuda()

            preds = self.model(input_vars)

            preds = preds.transpose(1,0)
            preds_size = torch.IntTensor(preds.size(1)).fill_(preds.size(0))

            targets = self.decoder.encode(target_vars, target_size)

            loss = self.criterion(preds, targets, preds_size, target_size)
            avg_loss += (loss.item() / input_vars.size(0))

            self.model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grade_norm)
            self.optimizer.step()

            del preds
            del loss

        return avg_loss / (idx_batch + 1), train_lr
