import os
import json

#######################################################################################################################

class Vocabulary(object):
    def __init__(self, root_dir):
        self.token2idx = {}
        self.idx2token = {}
        self.idx = 0
        self.path = os.path.join(root_dir, 'vocab.json')

        self.add(chr(1)) # extra output dimension for the BLANK label required for CTC loss

    def add(self, token):
        if not token in self.token2idx:
            self.token2idx[token] = self.idx
            self.idx2token[self.idx] = token
            self.idx += 1

    def __call__(self, val):
        res = None
        if isinstance(val, str):
            res = self.token2idx[val] if val in self.token2idx else self.token2idx['']
        elif isinstance(val, int):
            res = self.idx2token[val] if val <= self.__len__() else self.token2idx['']
        else:
            raise RuntimeError
        return res

    def __len__(self):
        return len(self.token2idx)

    def create(self, alphabet):
        for c in alphabet:
            self.add(c)

    def dump(self):
        data = {'idx': self.idx, 'token2idx':self.token2idx, 'idx2token':self.idx2token}
        with open(self.path, "w") as fd:
            json.dump(data, fd)

    def load(self):
        with open(self.path, "r") as fd:
            data = json.load(fd)
            self.idx = int(data['idx'])
            self.token2idx = data['token2idx']
            self.idx2token = {int(k):v for k,v in data['idx2token'].items()}

    def __repr__(self):
        return ''.join(list(self.token2idx.keys()))

#######################################################################################################################

if __name__ == '__main__':
    vocab = Vocabulary("./data/")

    # vocab.create('0123456789abcdefghijklmnopqrstuvwxyz')
    # vocab.dump()

    print(vocab)

    vocab.load()

    print("".join([vocab(i) for i in range(len(vocab))]))
