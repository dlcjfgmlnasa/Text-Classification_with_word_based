# -*- coding:utf-8 -*-
import os
import torch
import pickle
import argparse
import torch.nn as nn
import torch.optim as optim
from common import accuracy
from torch.utils.data import DataLoader
from data_helper import TextDataSet, Dictionary
from tensorboardX import SummaryWriter


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str,
                        default=os.path.join('./dataset', 'ratings_train.txt'))
    parser.add_argument('--dictionary_path', type=str,
                        default=os.path.join('./dataset', 'dictionary', 'word2idx.dic'))
    parser.add_argument('--model', type=str,
                        default='TextCNN',
                        choices=['TextCNN'])

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--out_channels', type=int, default=20)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--n_grams', type=list, default=[2, 3, 4])

    parser.add_argument('--print_summary_step', type=int, default=50)
    parser.add_argument('--print_step', type=int, default=50)

    parser.add_argument('--summary_store', type=str,
                        default=os.path.join('./store', 'text_cnn', 'log'))
    parser.add_argument('--model_store', type=str,
                        default=os.path.join('./store', 'text_cnn', 'model'))
    return parser.parse_args()


def get_model(model_type, param) -> torch.nn.Module:
    if model_type == 'TextCNN':
        from models.textcnn import TextCNN
        model = TextCNN(**param)
        return model
    else:
        raise NotImplemented()


def load_dictionary(dictionary_path):
    word2idx = pickle.load(open(dictionary_path, 'rb'))
    return word2idx


def data_loader(data_path: str, dictionary: Dictionary, seq_len: int, batch_size: int, shuffle_flag: bool) \
        -> DataLoader:
    # Dataset & DataLoader
    dataset = TextDataSet(
        data_path=data_path,
        dictionary=dictionary,
        seq_len=seq_len
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag
    )
    return dataloader


def main(argument):
    # Loading Dictionary
    if not os.path.exists(argument.dictionary_path):
        base_path = os.path.dirname(argument.dictionary_path)
        dictionary = Dictionary(
            data_path=argument.train_path,
            dict_path=base_path
        )
    else:
        dictionary = Dictionary()

    # Loading Word2Idx
    word2idx = dictionary.load_word2idx()
    word2idx_len = len(word2idx)

    # Tensorboard Summary
    summary = SummaryWriter(argument.summary_store)

    # Checking Model Store
    if not os.path.exists(argument.model_store):
        os.makedirs(argument.model_store)

    # Train Dataset & DataLoader
    train_dataloader = data_loader(argument.train_path, dictionary,
                                   seq_len=argument.seq_len, batch_size=argument.batch_size, shuffle_flag=True)

    # Get Model
    parameter = {
        'seq_len': argument.seq_len,
        'word_size': word2idx_len,
        'embedding_dim': argument.embedding_dim,
        'out_channels': argument.out_channels,
        'classes': argument.classes,
        'n_grams': argument.n_grams,
        'padding_idx': word2idx['__PAD__']
    }
    model = get_model(
        model_type=argument.model,
        param=parameter
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx['__PAD__'])

    print('Training Start...')

    total_count = 0
    for epoch in range(argument.epochs):
        epoch_count = 0
        average_loss = 0
        for i, data in enumerate(train_dataloader):
            feature, label = data

            output = model(feature)

            output = output.to(device=device)
            label = label.to(device=device, dtype=torch.int64)

            # Calculation Accuracy & Loss
            loss = criterion(input=output, target=label)
            acc = accuracy(inputs=output, target=label)

            # Optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Printing...
            # - Print Log
            if i % argument.print_step == 0:
                print('epoch : {0: 3d} \t i : {1:3d} \t loss : {2:5f} \t accuracy : {3:5f}'.
                      format(epoch, i, loss.item(), acc.item()))

            # - Print Summary
            if i % argument.print_summary_step == 0:
                summary.add_scalar('loss', loss.item(), total_count)
                summary.add_scalar('accuracy', acc.item(), total_count)

            average_loss += loss.item()
            epoch_count += 1
            total_count += 1

        # Average Loss...
        average_loss /= epoch_count

        # Saving model...
        model_path = os.path.join(argument.model_store, '{0:04d}.pth'.format(epoch))
        torch.save({
            'epoch': epoch,
            'seq_len': argument.seq_len,
            'model_type': argument.model,
            'dictionary_path': argument.dictionary_path,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': average_loss,
            'parameter': parameter
        }, model_path)

    print('Complete...')


if __name__ == '__main__':
    args = get_args()
    main(args)