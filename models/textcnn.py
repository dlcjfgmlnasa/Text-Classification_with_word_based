# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, seq_len, word_size, embedding_dim, out_channels, classes, n_grams, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=word_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        self.convolution_list = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=out_channels,
                kernel_size=n_gram,
                stride=1
            )
            for n_gram in n_grams
        ])
        self.max_pool_list = nn.ModuleList([
            nn.MaxPool1d(
                kernel_size=seq_len-n_gram+1,
                stride=1
            )
            for n_gram in n_grams
        ])

        self.linear = nn.Linear(
            in_features=out_channels * len(n_grams),
            out_features=classes
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Word Embedding
        embedded = self.embedding(inputs)
        embedded = embedded.permute(0, 2, 1)

        # Convolution Neural Network & Max Pooling
        cnn_output = []
        for convolution, max_pool in zip(self.convolution_list, self.max_pool_list):
            cnn_embedded = convolution(embedded)        # 1D-Convolution Neural Network
            cnn_embedded = torch.relu(cnn_embedded)     # Activation Function
            cnn_embedded = max_pool(cnn_embedded)       # 1D-Max Pooling
            cnn_embedded = cnn_embedded.squeeze(dim=2)
            cnn_output.append(cnn_embedded)
        cnn_output = torch.cat(cnn_output, dim=1)

        # Linear
        output = self.linear(cnn_output)
        return output
