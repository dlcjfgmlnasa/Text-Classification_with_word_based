# -*- coding:utf-8 -*-
import os
import torch
from data_helper import Dictionary, DataPreProcessing

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Predication(object):
    def __init__(self, model_store_path: str, model_store_name: str):
        print('Model Loading....')
        self.model_store = self.get_model_store(
            os.path.join(model_store_path, model_store_name)
        )
        self.dictionary = self.load_dictionary()
        self.dp = DataPreProcessing(
            dictionary=self.dictionary,
            seq_len=self.model_store['seq_len']
        )
        self.model = self.model_store['model'].to(device)
        print('Model Loading Complete!!')

    def load_dictionary(self) -> Dictionary:
        dictionary = Dictionary(
            dict_path=self.model_store['dictionary_path']
        )
        return dictionary

    def sentence_classification(self, sentence_list: list) -> list:
        total_vector = []
        for sentence in sentence_list:
            idx_list = self.dp.apply_word2idx(sentence)
            idx_list = self.dp.padding(idx_list)
            vector = torch.tensor(idx_list, device=device, dtype=torch.int64)
            total_vector.append(vector)

        total_vector = torch.stack(total_vector, dim=0).to(device=device, dtype=torch.int64)
        value, indicate = self.model(total_vector).max(dim=-1)
        result = list([int(i) for i in indicate])
        return result

    @staticmethod
    def get_model_store(store_path: str) -> dict:
        model_info = torch.load(store_path)
        if model_info['model_type'] == 'TextCNN':
            from models.textcnn import TextCNN
            model = TextCNN(**model_info['parameter'])

        elif model_info['model_type'] == 'TextRNN':
            from models.textrnn import TextRNN
            model = TextRNN(**model_info['parameter'])

        elif model_info['model_type'] == 'BiRNNWithAttention':
            from models.bi_rnn_with_attention import BiRNNWithAttention
            model = BiRNNWithAttention(**model_info['parameter'])

        else:
            raise NotImplemented()

        model.load_state_dict(model_info['model_state_dict'])
        result = {
            'model': model,
            'dictionary_path': model_info['dictionary_path'],
            'seq_len': model_info['seq_len']
        }
        return result
