# -*- coding:utf-8 -*-
import os
import json
import torch
import pickle
import pandas as pd
from konlpy.tag import Okt
from torch.utils.data import Dataset


class Dictionary(object):
    def __init__(self,
                 dict_path=os.path.join('./dataset', 'dictionary'),
                 min_count=0,
                 max_count=50000
                 ):
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

    def make_dict(self, data_path):
        df = pd.DataFrame([
            (line.split('\t')[1].strip(), line.split('\t')[2].strip())
            for line in open(data_path, 'r', encoding='utf-8').readlines()[1:]
        ], columns=['document', 'label'])
        df = df.dropna(how='any')
        df.document = df.document.str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
        data_size = len(df)

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
        for i, document in enumerate(df.document):
            if i % 100 == 0:
                print('\t\t{0:6d}/{1:6d} => {2:2.2f}%'.
                      format(i, data_size, (i / data_size) * 100))

            tokens = DataPreProcessing.split_tokenizer(document)
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


class DataPreProcessing(object):
    tokenizer = Okt()

    def __init__(self, dictionary: Dictionary, seq_len: int):
        self.dictionary = dictionary
        self.word2idx = self.dictionary.load_word2idx()
        self.seq_len = seq_len

    @staticmethod
    def split_tokenizer(line: str) -> list:
        # Split Tokenizer (document -> token)
        tokens = DataPreProcessing.tokenizer.morphs(line, norm=True, stem=True)
        return tokens

    def apply_word2idx(self, document: str) -> list:
        idx_list = []
        for token in DataPreProcessing.split_tokenizer(document):
            try:
                idx_list.append(self.word2idx[token])
            except KeyError:
                idx_list.append(self.word2idx['__UNK__'])
        return idx_list

    def padding(self, idx_list: list) -> list:
        size = len(idx_list)
        padding_idx = [self.word2idx['__PAD__'] for _ in range(self.seq_len - size)]
        idx_list = idx_list + padding_idx
        idx_list = idx_list[:self.seq_len]
        return idx_list

    def start(self, df, target_path, target_filename):
        data_size = len(df)

        if not os.path.exists(target_path):
            os.makedirs(target_path)
        target_path = os.path.join(target_path, target_filename)

        print('Start Preprocessing...')

        with open(target_path, 'w') as f:
            data = dict()
            for i, (document, label) in enumerate(zip(df.document, df.label)):
                # Print PreProcessing Progress
                if i % 100 == 0:
                    print('\t PreProcessing... => {0:6d}/{1:6d} - {2:2.2f}%'.
                          format(i, data_size, (i/data_size) * 100))

                feature = self.apply_word2idx(document)
                feature = self.padding(feature)

                data[i] = {
                    'feature': feature,
                    'label': int(label.strip())
                }
            json.dump(data, f, sort_keys=True)
        print('Complete...')


class TextDataSet(Dataset):
    def __init__(self, path, device: torch.device):
        self.dataset = json.load(open(path, 'r'))
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[str(index)]
        feature = torch.tensor(data['feature'], device=self.device, dtype=torch.int64)
        label = torch.tensor(data['label'], device=self.device, dtype=torch.int64)
        return feature, label
