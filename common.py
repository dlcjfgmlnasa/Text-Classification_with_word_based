# -*- coding:utf-8 -*-
import torch


def accuracy(inputs, target):
    # Calculation Accuracy
    values, indices = inputs.max(dim=-1)
    acc = torch.mean(torch.eq(indices, target).to(torch.float32))
    return acc
