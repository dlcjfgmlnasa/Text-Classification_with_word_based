# -*- coding:utf-8 -*-
import os
import torch
import argparse
import torch.nn as nn
from common import get_model_info
from torch.utils.data import DataLoader
from data_helper import Dictionary, TextDataSet
from common import accuracy


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str,
                        default=os.path.join('./dataset', 'ratings_test.txt'))
    parser.add_argument('--model_store_path', type=str,
                        default=os.path.join('./store', 'text_cnn', 'model', '0000.pth'))
    return parser.parse_args()


def main(argument):
    # Loading model
    model_info = get_model_info(argument.model_store_path)

    # Loading Dictionary
    if not os.path.exists(model_info['dictionary_path']):
        base_path = os.path.dirname(model_info['dictionary_path'])
        dictionary = Dictionary(
            data_path=argument.test_path,
            dict_path=base_path
        )
    else:
        dictionary = Dictionary()

    word2idx = dictionary.load_word2idx()

    # Test Dataset & DataLoader
    dataset = TextDataSet(data_path=argument.test_path, dictionary=dictionary, seq_len=model_info['seq_len'])
    dataloader = DataLoader(dataset=dataset, batch_size=1000, shuffle=False)

    criterion = nn.CrossEntropyLoss(ignore_index=word2idx['__PAD__'])

    print('Test Start...')

    total_count = 0
    with torch.no_grad():
        total_loss = 0
        total_accuracy = 0

        model = model_info['model']     # Get model
        for i, (feature, label) in enumerate(dataloader):
            output = model(feature)
            output = output.to(device=device)
            label = label.to(device=device, dtype=torch.int64)

            loss = criterion(input=output, target=label)
            acc = accuracy(inputs=output, target=label)

            print('i : {0:3d} \t loss : {1:5f} \t accuracy : {2:5f}'.format(i, loss.item(), acc.item()))
            total_accuracy += acc.item()
            total_loss += loss.item()
            total_count += 1

    # Calculation Loss & Accuracy
    total_loss /= total_count
    total_accuracy /= total_count
    print('total_loss : {0:5f} \t total_accuracy : {1:5f}'.format(total_loss, total_accuracy))


if __name__ == '__main__':
    args = get_args()
    main(args)
