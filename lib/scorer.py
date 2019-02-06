import Levenshtein as lev

#######################################################################################################################

class Scorer(object):
    def __init__(self):
        pass

    def score(self,predicted_vars, target_vars):
        idx, acc, cer = 0, 0.0, 0.0
        for idx, (dseq, tseq) in enumerate(zip(predicted_vars, target_vars)):
            acc += 1 if dseq == tseq else 0
            cer += lev.distance(dseq, tseq) / float(len(tseq))

        return acc / (idx + 1), cer / (idx + 1)

#######################################################################################################################

from difflib import SequenceMatcher

class ScorerSM(object):
    def __init__(self):
        pass

    def score(self, predicted_vars, target_vars):
        idx, acc, cer = 0, 0.0, 0.0
        for idx, (dseq, tseq) in enumerate(zip(predicted_vars, target_vars)):
            acc += SequenceMatcher(None, dseq, tseq).ratio()

        return (acc / (idx + 1)), 0

#######################################################################################################################

class ScorerLEV(object):
    def __init__(self):
            pass

    @staticmethod
    def wer(s1, s2):
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]
        return lev.distance(''.join(w1), ''.join(w2))

    @staticmethod
    def cer(s1, s2):
        s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
        return lev.distance(s1, s2)

    def score(self,predicted_vars, target_vars):
        idx, acc, cer = 0, 0.0, 0.0
        for idx, (dseq, tseq) in enumerate(zip(predicted_vars, target_vars)):
            acc += self.wer(dseq, tseq) / float(len(tseq.split()))
            cer += self.cer(dseq, tseq) / float(len(tseq))

        return 1 - (acc / (idx + 1)), cer / (idx + 1)


#######################################################################################################################
#  https://github.com/SeanNaren/deepspeech.pytorch/blob/master/decoder.py


class SpellingScorer(object):
    def __init__(self, speller):
        self.speller = speller

    def score(self, predicted_vars, target_vars):
        idx, acc, cer = 0, 0.0, 0.0
        for idx, (dseq, tseq) in enumerate(zip(predicted_vars, target_vars)):
            if dseq in self.speller.words:
                acc += 1 if dseq == tseq else 0
            else:
                dseq = self.speller.correction(dseq)
                acc += 1 if  dseq == tseq else 0

            cer += lev.distance(dseq, tseq) / float(len(tseq))

        return acc / (idx + 1), cer / (idx + 1)

#######################################################################################################################
