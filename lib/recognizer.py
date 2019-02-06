import torch
from torch.autograd import Variable

#######################################################################################################################

class Recognizer(object):
    def __init__(self, model, decoder):
        self.model = model
        self.decodr = decoder
        self.use_cuda = next(self.model.parameters()).is_cuda

    def recognize(self, image_vars):
        image_vars = Variable(image_vars)
        if self.use_cuda :
            image_vars = image_vars.cuda()

        self.model.eval()
        with torch.no_grad():

            preds = self.model(image_vars)
            preds = preds.transpose(1,0)

            decoder_vars = preds.max(2)[1].transpose(1, 0)
            decoder_seq = self.decodr.decode(decoder_vars.cpu(), remove_repetitions=True, filler='')

        return decoder_seq