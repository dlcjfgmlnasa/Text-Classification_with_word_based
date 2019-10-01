# -*- coding:utf-8 -*-
import os
import torch
import pickle
import numpy as np
import pandas as pd
from konlpy.tag import Okt
from torch.utils.data import Dataset


def split_tokenizer(line: str, tokenizer) -> list:
    tokens = tokenizer.morphs(line, norm=True)
    return tokens


class Dictionary(object):
    def __init__(self,
                 data_path=None,
                 dict_path=os.path.join('./dataset', 'dictionary'),
                 min_count=0,
                 max_count=50000
                 ):
        if data_path:
            self.df = pd.DataFrame([
                (line.split('\t')[1].strip(), line.split('\t')[2].strip())
                for line in open(data_path, 'r', encoding='utf-8').readlines()[1:]
            ], columns=['document', 'label'])

        self.SPK_KEY = ['__UNK__', '__PAD__', '__STR__', '__END__']
        self.word2count_name = 'word2count.dic'
        self.word2idx_name = 'word2idx.dic'
        self.idx2word_name = 'idx2word.dic'

        self.dict_path = dict_path
        self.min_count = min_count
        self.max_count = max_count
        self.tokenizer = Okt()

    def load_word2idx(self):
        word2idx_path = os.path.join(self.dict_path, self.word2idx_name)
        if os.path.exists(word2idx_path):
            word2idx = pickle.load(open(word2idx_path, 'rb'))
            return word2idx
        else:
            raise FileNotFoundError()

    def make_dict(self):
        print('make dictionary')
        if not os.path.exists(self.dict_path):
            os.makedirs(self.dict_path)

        # file pointer (word2count, word2idx, idx2word)
        word2count_fp = open(os.path.join(self.dict_path, self.word2count_name), 'wb')
        word2idx_fp = open(os.path.join(self.dict_path, self.word2idx_name), 'wb')
        idx2word_fp = open(os.path.join(self.dict_path, self.idx2word_name), 'wb')

        # Calculation Word2Count
        print('\t - 1. Calculation Word2Count')
        word2count = {}
        for line in self.df.document:
            tokens = split_tokenizer(line, self.tokenizer)
            for token in tokens:
                try:
                    word2count[token] += 1
                except KeyError:
                    word2count[token] = 1
        print('\t\t Complete...')

        # Remove min, max count word
        print('\t - 2. Remove words (min, max)')
        del_words = []
        for word, count in word2count.items():
            if not self.min_count <= count <= self.max_count:
                del_words.append(word)
        print('\t\t Complete...')

        # Sorting words
        print('\t - 3. word sorting')
        words = list(word2count.keys())
        words.sort()
        words = self.SPK_KEY + words
        print('\t\t Complete...')

        # Calculation Word2Idx, Idx2Word
        print('\t - 4. Calculation Word2Idx, Idx2Word')
        word2idx = {}
        idx2word = {}
        for i, word in enumerate(words):
            word2idx[word] = i
            idx2word[i] = word
        print('\t\t Complete...')

        # Saving word2count, word2idx, idx2word...
        print('\t - 5. Saving word2count, word2idx, idx2word')
        pickle.dump(word2count, word2count_fp)
        pickle.dump(word2idx, word2idx_fp)
        pickle.dump(idx2word, idx2word_fp)
        print('\t\t Complete...')


class TextDataSet(Dataset):
    def __init__(self, data_path: str, dictionary: Dictionary, seq_len: int):
        self.df = pd.DataFrame([
            (line.split('\t')[1].strip(), line.split('\t')[2].strip())
            for line in open(data_path, 'r', encoding='utf-8').readlines()[1:]
        ], columns=['document', 'label'])
        self.df.label = self.df.label.astype(np.int)

        self.dictionary = dictionary
        self.word2idx = self.dictionary.load_word2idx()
        self.seq_len = seq_len
        self.tokenizer = Okt()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        data = self.df.iloc[index]
        feature = torch.tensor(self.padding(self.apply_word2idx(data.document)))
        label = torch.tensor(data.label)
        return feature, label

    def apply_word2idx(self, line):
        idx_list = []
        for token in split_tokenizer(line, tokenizer=self.tokenizer)[:self.seq_len]:
            try:
                idx_list.append(self.word2idx[token])
            except KeyError:
                idx_list.append(self.word2idx['__UNK__'])
        return idx_list

    def padding(self, idx_list):
        size = len(idx_list)
        padding_idx = [self.word2idx['__PAD__'] for _ in range(self.seq_len - size)]
        idx_list = idx_list + padding_idx
        return idx_list
