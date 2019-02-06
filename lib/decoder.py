import torch
import torch.nn.functional as F


#######################################################################################################################

class CTCGreedyDecoder(object):
    def __init__(self, vocabulary):
        self.vocab = vocabulary

    @staticmethod
    def encode(target_variables, target_size):
        targets = [t[0][0:t[1]] for t in zip(target_variables, target_size)]
        targets = torch.cat(targets, 0).type(torch.IntTensor)
        return targets

    def decode(self, input_vars, input_size, remove_repetitions=False, filler=''):

        decoded_seq = []
        for seq, seq_len in zip(input_vars.cpu(), input_size.cpu()):
            txt = ''
            for i in range(seq_len):
                if remove_repetitions and seq[i] and i > 0 and seq[i - 1] == seq[i]:
                    continue
                c = seq[i].item()
                txt += self.vocab(c) if c else filler

            decoded_seq.append(txt)
        return decoded_seq

    def __call__(self, input_vars, target_vars, input_size, target_size):
        # input_vars : seqLength x batchSize x alphabet_size
        # input_size : batchSize

        decoder_vars = input_vars.max(2)[1].transpose(1, 0)
        decoder_size = input_size.cpu()

        decoder_seq = self.decode(decoder_vars, decoder_size, remove_repetitions=True, filler='')

        target_seq = None
        if target_vars is not None:

            target_vars = target_vars.cpu()
            target_size = target_size.cpu()

            target_seq = self.decode(target_vars, target_size)

        return decoder_seq, target_seq

#######################################################################################################################

# class CTCBeamSearchDecoder(object):
#     def __init__(self, vocabulary, lm_path=None, alpha=0, beta=0, cutoff_top_n=40, cutoff_prob=1.0, beam_width=20,
#                  num_processes=8, blank_index=0):
#         self.vocab = vocabulary
#         self.vocab_list = list(self.vocab.token2idx.keys())
#
#         self.ctcdecoder = CTCBeamDecoder(self.vocab_list,lm_path, alpha, beta, cutoff_top_n, cutoff_prob, beam_width,
#                                          num_processes, blank_index)
#
#     @staticmethod
#     def encode(target_variables, target_size):
#         targets = [t[0][0:t[1]] for t in zip(target_variables, target_size)]
#         targets = torch.cat(targets, 0).type(torch.IntTensor)
#         return targets
#
#     def decode(self, input_vars, input_size):
#
#         decoded_seq = []
#         for seq, seq_len in zip(input_vars, input_size):
#             txt = ''.join([self.vocab(x.item()) for x in seq[0:seq_len]])
#             decoded_seq.append(txt)
#         return decoded_seq
#
#     def __call__(self, input_vars, target_vars, input_size, target_size):
#         # input_vars : seqLength x batchSize x alphabet_size
#         # input_size : batchSize
#
#         probs_seq = input_vars.cpu().data.transpose(0, 1).contiguous()
#         probs_size = input_size.cpu()
#
#         probs_seq = F.softmax(probs_seq, dim=-1)
#
#         out_seq, _, _, out_seq_len = self.ctcdecoder.decode(probs_seq, probs_size)
#
#         decoder_seq = self.decode(out_seq[:,0,:], out_seq_len[:,0])
#
#         target_vars = target_vars.cpu()
#         target_size = target_size.cpu()
#
#         target_seq = self.decode(target_vars, target_size)
#
#         return decoder_seq, target_seq
