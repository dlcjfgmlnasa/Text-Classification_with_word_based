# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as f


class TextRNN(nn.Module):
    def __init__(self, word_size: int, embedding_dim: int, rnn_dim: int, num_layer: int,
                 classes: int, padding_idx: int, bidirectional: bool):
        super().__init__()
        self.embedding = nn.Embedding(
            embedding_dim=embedding_dim,
            num_embeddings=word_size,
            padding_idx=padding_idx
        )
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=rnn_dim,
            num_layers=num_layer,
            bidirectional=bidirectional,
            batch_first=True
        )
        if bidirectional:
            output_size = rnn_dim * 2
        else:
            output_size = rnn_dim

        self.fc = nn.Linear(
            in_features=output_size,
            out_features=classes
        )

    def forward(self, inputs):
        # Word Embedding
        embedded = self.embedding(inputs)       # => (batch_size, seq_len, embedded_size)

        # LSTM Neural Network
        rnn_output, _ = self.rnn(embedded)
        last_rnn_output = rnn_output[:, -1, :]

        # Fully Connection Neural Network
        output = self.fc(last_rnn_output)
        output = f.softmax(output, dim=-1)
        return output
