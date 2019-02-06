import os
import pickle
import gensim


#######################################################################################################################
# https://www.kaggle.com/cpmpml/spell-checker-using-word2vec

class Spelling(object):
    def __init__(self, root, words=None):
        self.WORDS = []
        self.root = root
        self.words = words if not words is None else []

        self.path = os.path.join(self.root, 'GoogleNews-vectors-negative300.words')
        with open( self.path, "rb" ) as fb:
            self.WORDS = pickle.load(fb)

    @classmethod
    def create(cls, path='./data/GoogleNews-vectors-negative300.bin'):
        model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
        words = model.index2word

        w_rank = {}
        for i,word in enumerate(words):
            w_rank[word] = i

        WORDS = w_rank

        with open( "./data/GoogleNews-vectors-negative300.words", "wb" ) as fb:
            pickle.dump(WORDS, fb)

    def correction(self, word):
        """Most probable spelling correction for word."""

        def P(word):
            """Probability of `word`."""
            # use inverse of rank as proxy
            # returns 0 if the word isn't in the dictionary
            return - self.WORDS.get(word, 0)

        return max(self.candidates(word), key=P)

    def candidates(self, word):
        """Generate possible spelling corrections for word."""
        return self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word]

    def known(self, words):
        """The subset of `words` that appear in the dictionary of WORDS."""
        return set(w for w in words if w in self.WORDS)

    @staticmethod
    def edits1(word):
        """All edits that are one edit away from `word`."""
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        """All edits that are two edits away from `word`."""
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Word File:    {}\n'.format(self.path)
        fmt_str += '    Size:         {}\n'.format(len(self.WORDS))
        return fmt_str

#######################################################################################################################

if __name__ == '__main__':
    sp = Spelling("./data")

    print(sp)

    print(sp.correction('antoracy')  )

