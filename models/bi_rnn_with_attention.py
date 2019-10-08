# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class BiRNNWithAttention(nn.Module):
    def __init__(self, word_size: int, embedding_dim: int, rnn_dim: int, num_layer: int,
                 classes: int, padding_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(embedding_dim=embedding_dim, num_embeddings=word_size,
                                      padding_idx=padding_idx)

        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=rnn_dim,
                           num_layers=num_layer, bidirectional=True, batch_first=True)

        # bi concat fully connection neural network
        self.bi_concat_output = nn.Linear(in_features=rnn_dim*2, out_features=rnn_dim)

        # fully connected neural network
        self.fc = nn.Linear(in_features=rnn_dim, out_features=classes)

    def forward(self, inputs):
        embedded = self.embedding(inputs)

        # LSTM model
        rnn_output, (final_hidden_state, final_cell_state) = self.rnn(embedded)

        # concatenate output & final_hidden_state
        rnn_output = self.bi_concat_output(rnn_output)
        hidden_state = torch.sum(final_hidden_state, dim=0).unsqueeze(dim=-1)

        # attention mechanism
        attention_weight = rnn_output.bmm(hidden_state)
        attention_score = nn.Softmax(dim=1)(attention_weight)
        context_vector = rnn_output * attention_score
        context_vector = context_vector.sum(dim=1)

        # fully connected neural network
        output = self.fc(context_vector)
        output = nn.Softmax(dim=-1)(output)
        return output
