import os
from glob import glob
from os.path import basename

import json
import random
import numpy as np
import re
import math

import PIL
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import tensorlayer.prepro as prepro

from .vocabulary import Vocabulary
from .utilities import HYPERPARAMETERS

#######################################################################################################################

class IAMHandwritingDataset(data.Dataset):
    """IAM Handwriting Database ( http://www.fki.inf.unibe.ch/databases/iam-handwriting-database )"""

    def __init__(self, root, vocab, dataset, transform=None, target_transform=None, max_size=None):
        self.root = root
        self.vocab = vocab
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.max_size = max_size


        self.annotations = []
        self.max_seq_length = 0
        self.index = []

        self.annotations, self.max_seq_length = self.load_annotations(root, dataset)
        self.index = self.load_index(root)

        if isinstance(self.max_size, int):

            assert self.max_size < len(self.annotations) # max_train_size needs to select a subset

            self.annotations = random.sample(self.annotations, self.max_size)

    def __getitem__(self, index):
        annotation = self.annotations[index]
        line_id = annotation['line_id']
        path = self.index[line_id]

        image = Image.open(path).convert('L')
        if self.transform is not None:
            image = self.transform(image)

        target = annotation['transcript']
        target = [self.vocab(token) for token in target]
        target = np.array(target)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.annotations)

    @classmethod
    def create(cls, root):
        path = os.path.join(root, 'part/lines/aachen/tr.lst')
        with open(path, 'r') as infile:
            trainset = {v.strip() for v in infile.readlines()}

        path = os.path.join(root, 'part/lines/aachen/va.lst')
        with open(path , 'r') as infile:
            validationset1 = {v.strip() for v in infile.readlines()}

        try:
            path = os.path.join(root, 'part/lines/aachen/va2.lst')
            with open(path, 'r') as infile:
                validationset2 = {v.strip() for v in infile.readlines()}
        except:
            validationset2 = {}

        path = os.path.join(root, 'part/lines/aachen/te.lst')
        with open(path, 'r') as infile:
            testset = {v.strip() for v in infile.readlines()}

        file_index = cls.load_index(root)

        path = os.path.join(root, 'ascii/lines.txt')

        annotations = []
        max_seq_length = 0
        with open(path, "r") as fd:
            for idx, line in enumerate(fd.readlines()):
                if '#' in line[0]:
                    continue
                fields = line.rstrip().split(" ")

                transcript = fields[8]
                # transcript field has whitespaces in it - need to fix the erroneous splits
                if len(fields) > 9:
                    for i in range(9, len(fields)):
                        transcript += fields[i]

                # replace | seperator with whitespace
                transcript = transcript.replace("|", " ")

                l = len(transcript)
                if not l:
                    continue

                if l > max_seq_length:
                    max_seq_length = l

                annotation={
                    'line_id': fields[0],
                    'seg_result': fields[1],
                    'graylevel': fields[2],
                    'components': fields[3],
                    'bounding_box':[fields[4], fields[5], fields[6], fields[7]],
                    'transcript': transcript
                }

                line_id = annotation['line_id']
                path = file_index[line_id]
                try:
                    _ = Image.open(path)
                except IOError:
                    print('Corrupted image for %s ' % path)
                    continue

                if not line_id in file_index:
                    print('File missing %s' % path)
                    continue

                line_id = annotation['line_id']#[:-4]

                if line_id in trainset:
                    dataset = "train"
                elif line_id in validationset1:
                    dataset = "valid1"
                elif line_id in validationset2:
                    dataset = "valid2"
                elif line_id in testset:
                    dataset = "test"
                else:
                    dataset = "train"

                annotation['dataset'] = dataset

                annotations.append(annotation)

        path = os.path.join(root, 'annotations.json')

        with open(path, 'w') as outfile:
            json.dump({'annotations': annotations,
                       'max_seq_length': max_seq_length},
                      outfile)

    @classmethod
    def load_annotations(cls, root, dataset):
        path = os.path.join(root, 'annotations.json')

        with open(path, "r") as fd:
            content = json.load(fd)

        if not len(content):
            raise RuntimeError('Dataset empty.')

        annotations = content['annotations']
        max_seq_length = content['max_seq_length']

        if "train" in dataset:
            annotations = [a for a in annotations if 'train' in a['dataset']]
        elif "valid" in dataset:
            annotations = [a for a in annotations if 'valid1' in a['dataset'] or 'valid2' in a['dataset']]
        elif "test" in dataset:
            annotations = [a for a in annotations if 'test' in a['dataset']]
        else:
            raise RuntimeError('Dataset parameter can be: "train", "valid" or "test".')

        return annotations, max_seq_length

    @classmethod
    def load_index(cls, root):
        path = os.path.join(root, 'imgs/lines_h64/') + "*.jpg"
        files = glob(path)

        index = dict([(basename(path).rstrip(".jpg") , path) for path in files])

        if not len(index):
            raise RuntimeError('Image files empty.')
        return index

    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Number of images found: {}\n'.format(len(self.index))
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

#######################################################################################################################

class FromNumpyToTensor(object):
    def __call__(self, arr):
        return torch.LongTensor(arr)
    def __repr__(self):
        return self.__class__.__name__ + '()'

#######################################################################################################################

class ResizeAndPad(object):

    def __init__(self, h_size, max_width):
        self.h_size = h_size
        self.max_width = max_width

    def __call__(self, img):
        w, h = img.size

        hratio = self.h_size / float(h)
        w_size = int((float(w) * float(hratio)))

        res_img = img.resize((w_size, self.h_size), Image.ANTIALIAS)

        w_delta = self.max_width - w_size
        pad_img = PIL.ImageOps.expand(res_img, (0,0,w_delta,0), 0)

        return pad_img

#######################################################################################################################

class DataArgumentation(object):
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, img):
        if random.random() < self.threshold:
            img = np.array(img)
            img = img.reshape(1, img.shape[0], img.shape[1])

            img = prepro.rotation(img, 5, is_random=True, row_index=1, col_index=2, channel_index=0)
            img = prepro.shift(img, 0.02, 0.04, is_random=True, row_index=1, col_index=2, channel_index=0)
            img = prepro.shear(img, intensity=0.1, is_random=True, row_index=1, col_index=2, channel_index=0)
            img = prepro.zoom(img, (1.0, 1.2), is_random=True, row_index=1, col_index=2, channel_index=0)

            img = prepro.elastic_transform(img.squeeze(0), is_random=True, alpha=5.5, sigma=35, cval=0.0)

            img = Image.fromarray(img)
        return img

#######################################################################################################################

class alignCollate(object):

    def __init__(self, img_height=32):
        self.img_height = img_height

    def __call__(self, batch):
        images, targets = zip(*batch)

        sizes = [(image.size(1),image.size(2)) for image in images]
        widths = [ int(float(s[1]) * (self.img_height / float(s[0]))) for s in sizes]
        max_width = max(max(widths), self.img_height)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            ResizeAndPad(self.img_height, max_width),
            transforms.ToTensor()
        ])

        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)
        image_sizes = torch.IntTensor(widths)

        # Merge targets (from tuple of 1D tensor to 2D tensor).
        lengths = [len(target) for target in targets]
        out_lengths = torch.IntTensor(lengths)

        out_targets = torch.zeros(len(targets), max(lengths)).long()
        for i, target in enumerate(targets):
            end = lengths[i]
            out_targets[i, :end] = target[:end]

        return images, out_targets, image_sizes, out_lengths

#######################################################################################################################
if __name__ == '__main__':

    H = HYPERPARAMETERS({
        'BATCH_SIZE'          : 16,
        'HEIGHT'              : 64,
        'NUM_WORKERS'         : 8,
        'HIDDEN_SIZE'         : 256,
        'NUM_LAYERS'          : 2,
        'RNN_DROPOUT'         : 0.5,
        'LR'                  : 0.0003,
        'LR_LAMBDA'           : lambda epoch : max( math.pow(0.78, math.floor((1 + epoch) / 4.0)), 0.1),
        'WEIGHT_DECAY'        : 0,
        'MAX_GRAD_NORM'       : 5.,
        'ARGUMENTATION'       : 0.7,
        'STOPPING_PATIENCE'   : 10,
        'NUM_EPOCHS'          : 10,
        'CHECKPOINT_FILE'     : 'IAM_Handwriting_Recognition_CRNN_Final_Version',
        'CHECKPOINT_INTERVAL' : 10,
        'CHECKPOINT_RESTORE'  : False ,
        'USE_CUDA'            : torch.cuda.is_available(),
    })

   # IAMHandwritingDataset.create("./data")

    vocab = Vocabulary("./data")
    vocab.load()
    print(vocab)


    image_transform_train = transforms.Compose([
        transforms.Pad(2, fill=0),
        DataArgumentation(threshold=0.7),
        transforms.ToTensor(),
    ])

    image_transform_test = transforms.Compose([
        transforms.Pad(2, fill=0),
        transforms.ToTensor(),
    ])

    target_transform = transforms.Compose([
        FromNumpyToTensor()
    ])

    train_dataset = IAMHandwritingDataset('./data', vocab, dataset="train",
                                          transform=image_transform_train, target_transform=target_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, num_workers=6, shuffle=True,
        collate_fn=alignCollate(img_height=64), pin_memory=True)

    print(train_dataset)
    print(len(train_loader))


    valid_dataset = IAMHandwritingDataset('./data', vocab, dataset="valid",
                                          transform=image_transform_test, target_transform=target_transform)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=H.BATCH_SIZE, shuffle=False, num_workers=H.NUM_WORKERS,
        collate_fn=alignCollate(img_height=H.HEIGHT), pin_memory=True)

    print(valid_dataset)
    print(len(valid_loader))

    test_dataset = IAMHandwritingDataset('./data', vocab, dataset="test",
                                         transform=image_transform_test, target_transform=target_transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=H.BATCH_SIZE, shuffle=False, num_workers=H.NUM_WORKERS,
        collate_fn=alignCollate(img_height=H.HEIGHT))

    print(test_dataset)
    print(len(test_loader))

    input_vars, target_vars, input_sizes, target_len = next(train_loader.__iter__())
