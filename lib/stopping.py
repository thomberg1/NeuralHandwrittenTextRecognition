


#######################################################################################################################

class Stopping(object):
    """
    Class implement some of regularization techniques to avoid over-training as described in
    http://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf
    """
    def __init__(self, model, patience=50):
        self.model = model
        self.patience = patience

        self.best_score = -1
        self.best_score_epoch = 0
        self.best_score_state = None

    def step(self, epoch, train_score, valid_score):
        if valid_score > self.best_score:
            self.best_score = valid_score
            self.best_score_epoch = epoch
            self.best_score_state = self.model.state_dict()
            return False
        elif self.best_score_epoch + self.patience < epoch:
            return True

    def state_dict(self):
        return {
            'patience' : self.patience,
            'best_score' : self.best_score,
            'best_score_epoch' : self.best_score_epoch,
        }

    def load_state_dict(self, state_dict):
        self.patience = state_dict['patience']
        self.best_score  = state_dict['best_score']
        self.best_score_epoch = state_dict['best_score_epoch']

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Patience: {}\n'.format(self.patience)
        fmt_str += '    Best Score: {:.4f}\n'.format(self.best_score)
        fmt_str += '    Epoch of Best Score: {}\n'.format(self.best_score_epoch)
        return fmt_str

#######################################################################################################################