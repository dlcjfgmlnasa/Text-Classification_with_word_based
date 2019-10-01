# -*- coding:utf-8 -*-
import torch


def accuracy(inputs, target):
    # Calculation Accuracy
    values, indices = torch.max(inputs, dim=1)
    acc = torch.mean(torch.eq(indices, target).to(dtype=torch.float32))
    return acc


def get_model_info(store_path: str) -> dict:
    # Get model information (for test.py, predication.py)
    model = None
    model_info = torch.load(store_path)
    if model_info['model_type'] == 'TextCNN':
        from models.textcnn import TextCNN
        model = TextCNN(**model_info['parameter'])
        model.load_state_dict(model_info['model_state_dict'])

    result = {
        'model': model,
        'dictionary_path': model_info['dictionary_path'],
        'seq_len': model_info['seq_len']
    }
    return result
