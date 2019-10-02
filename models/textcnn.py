# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, seq_len: int, word_size: int, embedding_dim: int,
                 out_channels: int, classes: int, n_grams: list, padding_idx: int, dropout_rate=0.5):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=word_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        self.convolutions = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=out_channels,
                    kernel_size=n_gram,
                    stride=1
                ),
                nn.ReLU(),
                nn.MaxPool1d(
                    kernel_size=seq_len-n_gram+1,
                    stride=1
                )
            )
            for n_gram in n_grams
        ])
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(
            in_features=out_channels * len(n_grams),
            out_features=classes
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Word Embedding
        embedded = self.embedding(inputs)
        embedded = embedded.permute(0, 2, 1)    # => (batch_size, embedded_size, seq_len)

        # Convolution Neural Network & Max Pooling
        cnn_output = []
        for cnn_sequence in self.convolutions:
            # 1D-Convolution Neural Network -> ReLU -> 1D Max Pooling
            cnn_embedded = cnn_sequence(embedded)
            cnn_embedded = cnn_embedded.squeeze(dim=2)
            cnn_output.append(cnn_embedded)

        # Convolution Neural Network Output Concat & Dropout
        flatten_cnn_output = torch.cat(cnn_output, dim=-1)
        flatten_cnn_output_dropout = self.dropout(flatten_cnn_output)

        # Fully Connection Neural Network
        output = self.fc(flatten_cnn_output_dropout)
        output = torch.sigmoid(output)

        return output
