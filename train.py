# -*- coding:utf-8 -*-
import os
import torch
import argparse
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from common import accuracy
from torch.utils.data import DataLoader
from data_helper import TextDataSet, Dictionary, DataPreProcessing
from tensorboardX import SummaryWriter


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_train_path', type=str,
                        default=os.path.join('./dataset', 'ratings_train.txt'))
    parser.add_argument('--dictionary_path', type=str,
                        default=os.path.join('./dataset', 'dictionary'))
    parser.add_argument('--data_preprocessing_file_dir', type=str,
                        default=os.path.join('./dataset', 'preprocessing'))

    parser.add_argument('--model', type=str,
                        default='TextRNN',
                        choices=['TextCNN', 'TextRNN'])

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--data_split_rate', type=float, default=0.2)

    # TextCNN
    parser.add_argument('--out_channels', type=int, default=50)
    parser.add_argument('--dropout_rate', type=float, default=0.8)
    parser.add_argument('--n_grams', type=list, default=[2, 3, 4])

    # TextRNN
    parser.add_argument('--rnn_dim', type=int, default=50)
    parser.add_argument('--rnn_num_layer', type=int, default=2)
    parser.add_argument('--rnn_bidirectional', type=bool, default=True)

    parser.add_argument('--print_summary_step', type=int, default=50)
    parser.add_argument('--print_step', type=int, default=50)
    parser.add_argument('--print_val_step', type=int, default=300)

    parser.add_argument('--summary_store', type=str,
                        default=os.path.join('./store', 'text_rnn', 'log'))
    parser.add_argument('--model_store', type=str,
                        default=os.path.join('./store', 'text_rnn', 'model'))
    return parser.parse_args()


class Trainer(object):
    def __init__(self, argument):
        self.argument = argument
        self.dictionary = self.load_dictionary()
        self.word2idx = self.dictionary.load_word2idx()
        self.dp = self.data_preprocessing()
        self.model = self.get_model()
        self.summary = SummaryWriter(self.argument.summary_store)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.argument.learning_rate)
        self.train()

    def train(self):
        # Training Start
        self.model.train()
        self.model.to(device)

        # Checking Model Store
        if not os.path.exists(self.argument.model_store):
            os.makedirs(self.argument.model_store)

        print('Training Start...')

        total_count = 0
        for epoch in range(self.argument.epochs):
            epoch_count = 0
            average_loss = 0
            average_accuracy = 0
            for i, data in enumerate(self.train_dataloader()):
                feature, label = data
                # NetWork Forward
                output = self.model(feature)

                # Calculation Accuracy & Loss
                loss, acc = self.get_accuracy_loss(inputs=output, target=label)

                # Optimizer & Network Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Printing...
                # - Print Log
                if i % self.argument.print_step == 0:
                    print('[Train] epoch : {0: 3d} \t i : {1:3d} \t loss : {2:2.6f} \t accuracy : {3:.5f}'.
                          format(epoch, i, loss.item(), acc.item()))

                # - Print Summary
                if i % self.argument.print_summary_step == 0:
                    self.summary.add_scalar('train/loss', loss.item(), total_count)
                    self.summary.add_scalar('train/accuracy', acc.item(), total_count)

                # - Calculation Validation
                if i % self.argument.print_val_step == 0:
                    val_loss, val_acc = self.val()
                    print('[ Val ] epoch : {0: 3d} \t i : {1:3d} \t loss : {2:2.6f} \t accuracy : {3:.5f}'.
                          format(epoch, i, loss.item(), acc.item()))
                    self.summary.add_scalar('val/loss', val_loss, total_count)
                    self.summary.add_scalar('val/accuracy', val_acc, total_count)

                average_loss += loss.item()
                epoch_count += 1
                total_count += 1

            # Average Loss & Accuracy
            average_loss /= epoch_count
            average_accuracy /= epoch_count

            # Saving model...
            self.save_model(
                epoch=epoch,
                average_loss=average_loss,
                average_accuracy=average_loss
            )

    def val(self):
        count = 0
        average_loss = 0
        average_acc = 0

        with torch.no_grad():
            for feature, label in self.val_dataloader():
                output = self.model(feature)
                loss, acc = self.get_accuracy_loss(inputs=output, target=label)
                average_loss += loss.item()
                average_acc += acc.item()
                count += 1
        average_loss /= count
        average_acc /= count
        return average_loss, average_acc

    def get_accuracy_loss(self, inputs, target):
        target_one_hot = f.one_hot(target, num_classes=self.argument.classes).float()

        # Calculation Accuracy & Loss
        loss = self.criterion(input=inputs, target=target_one_hot)
        acc = accuracy(inputs=inputs, target=target)
        return loss, acc

    def train_dataloader(self):
        # Train DataSet & DataLoader
        train_dataset = TextDataSet(
            path=os.path.join(self.argument.data_preprocessing_file_dir, 'preprocessing_train.json'),
            device=device
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.argument.batch_size,
            shuffle=True
        )
        return train_dataloader

    def val_dataloader(self):
        # Validation Dataset & DataLoader
        val_dataset = TextDataSet(
            path=os.path.join(self.argument.data_preprocessing_file_dir, 'preprocessing_val.json'),
            device=device
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=1000,
            shuffle=False
        )
        return val_dataloader

    def data_preprocessing(self):
        # Loading Dataset & Split Train / Validation
        frame = self.load_dataset()
        train_df, val_df = self.split_train_val_dataset(frame)

        # Get DataPreProcessing Class
        dp = DataPreProcessing(
            dictionary=self.dictionary,
            seq_len=self.argument.seq_len
        )
        if not os.path.exists(self.argument.data_preprocessing_file_dir):
            # if not exist preprocessing folder
            # => Train
            dp.start(
                df=train_df,
                target_path=self.argument.data_preprocessing_file_dir,
                target_filename='preprocessing_train.json',
            )
            # => Val
            dp.start(
                df=val_df,
                target_path=self.argument.data_preprocessing_file_dir,
                target_filename='preprocessing_val.json',
            )
        return dp

    def load_dataset(self):
        # Loading Original Dataset
        dataset = pd.DataFrame([
            (line.split('\t')[1], line.split('\t')[2])
            for line in open(self.argument.origin_train_path, 'r', encoding='utf-8').readlines()[1:]
        ], columns=['document', 'label'])
        dataset = dataset.dropna(how='any')
        dataset.document = dataset.document.str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
        return dataset

    def split_train_val_dataset(self, frame):
        # Split Train / Validation
        split_rate = self.argument.data_split_rate
        split_point = int(len(frame) * (1 - split_rate))
        train_df = frame.iloc[:split_point]
        val_df = frame.iloc[split_point:]
        return train_df, val_df

    def load_dictionary(self):
        # Loading Dictionary
        dictionary_path = self.argument.dictionary_path
        if not os.path.exists(dictionary_path):
            # if not exist dictionary (word2count, word2idx, idx2word)
            dictionary = Dictionary(
                dict_path=dictionary_path
            )
            dictionary.make_dict(self.argument.origin_train_path)
        else:
            dictionary = Dictionary(
                dict_path=self.argument.dictionary_path
            )
        return dictionary

    def get_model(self):
        model_type = self.argument.model
        parameter = self.get_parameter()
        if model_type == 'TextCNN':
            from models.textcnn import TextCNN
            return TextCNN(**parameter)
        elif model_type == 'TextRNN':
            from models.textrnn import TextRNN
            return TextRNN(**parameter)
        else:
            raise NotImplemented()

    def get_parameter(self):
        argument = self.argument
        model_type = argument.model
        if model_type == 'TextCNN':
            parameter = {
                'seq_len': argument.seq_len, 'word_size': len(self.word2idx),
                'embedding_dim': argument.embedding_dim, 'out_channels': argument.out_channels,
                'classes': argument.classes, 'n_grams': argument.n_grams,
                'padding_idx': self.word2idx['__PAD__'], 'dropout_rate': argument.dropout_rate
            }
            return parameter
        elif model_type == 'TextRNN':
            parameter = {
                'word_size': len(self.word2idx), 'embedding_dim': argument.embedding_dim, 'rnn_dim': argument.rnn_dim,
                'num_layer': argument.rnn_num_layer, 'classes': argument.classes,
                'padding_idx': self.word2idx['__PAD__'], 'bidirectional': argument.rnn_bidirectional
            }
            return parameter

    def save_model(self, epoch, average_loss, average_accuracy):
        # Saving model...
        model_path = os.path.join(self.argument.model_store, '{0:04d}.pth'.format(epoch))
        torch.save({
            'epoch': epoch,
            'seq_len': self.argument.seq_len,
            'model_type': self.argument.model,
            'dictionary_path': self.argument.dictionary_path,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch_average_loss': average_loss,
            'epoch_average_accuracy': average_accuracy,
            'parameter': self.get_parameter()
        }, model_path)


if __name__ == '__main__':
    args = get_args()
    Trainer(args)
