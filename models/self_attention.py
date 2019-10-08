# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, word_size: int, embedding_dim: int, num_units: int, num_heads: int,
                 classes: int, padding_idx: int):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=word_size, embedding_dim=embedding_dim,
                                       padding_idx=padding_idx)
        self.Q_linear = nn.Linear(in_features=embedding_dim, out_features=num_units)    # query
        self.K_linear = nn.Linear(in_features=embedding_dim, out_features=num_units)    # key
        self.V_linear = nn.Linear(in_features=embedding_dim, out_features=num_units)    # value
        self.num_units = num_units
        self.num_heads = num_heads

        # fully connection
        self.fc = nn.Linear(in_features=num_units, out_features=classes)

    def forward(self, inputs):
        embedded = self.embeddings(inputs)

        q = self.Q_linear(embedded)
        k = self.K_linear(embedded)
        v = self.V_linear(embedded)

        # split multi head
        q_ = torch.cat(torch.split(q, split_size_or_sections=self.num_heads, dim=-1), dim=0)
        k_ = torch.cat(torch.split(k, split_size_or_sections=self.num_heads, dim=-1), dim=0)
        v_ = torch.cat(torch.split(v, split_size_or_sections=self.num_heads, dim=-1), dim=0)

        # attention mechanism
        output = q_.matmul(k_.permute(0, 2, 1))
        attention_score = output / (k_.shape[1] ** 0.5)     # scale
        attention_distribution = nn.Softmax(dim=-1)(attention_score)
        output = torch.matmul(attention_distribution, v_)

        # reshape
        output = torch.cat(
            torch.split(output, split_size_or_sections=int(output.shape[0]/self.num_heads), dim=0), dim=-1)

        # residual connection (output = output + query)
        output += q

        # fully connected connection
        output = output.sum(dim=1)
        output = self.fc(output)
        output = nn.Softmax(dim=-1)(output)
        return output
