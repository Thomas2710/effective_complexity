import torch
from tqdm import tqdm
import numpy as np
import pandas as pd


def dataset_cifar(hyperparams):
    from effective_complexity.datasets.cifar import dataset_cifar as df
    return df(hyperparams)

def dataset_synthetic(hyperparams):
    from effective_complexity.datasets.synthetic import dataset_synthetic as ds
    return ds(hyperparams)

def dataset_penntreebank(hyperparams):
    from effective_complexity.datasets.treebank import dataset_penntreebank as dpt
    return dpt(hyperparams)

def list_datasets():
    datasets = [n.replace('dataset_', '')  for n, m in globals().items()
            if n.startswith('dataset_')]
    return datasets


def get_dataset_class(dataset_name):
    dataset = [m  for n, m in globals().items()
        if n.startswith('dataset_'+dataset_name)]

    return dataset

def collate_fn(batch):
    """ Custom function to collate dictionary-based data. """
    inputs = torch.stack([item["x"] for item in batch])  # Stack 3D tensors
    labels = torch.stack([item["label"] for item in batch])  # Convert labels to tensor
    return {"x": inputs, "label": labels}
