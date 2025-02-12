from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms, utils
from effective_complexity.datasets import collate_fn
import pandas as pd

def dataset_penntreebank(hyperparams):
    class PENNTREEBANK(Dataset):
        def __init__(self, num_sentences=100):
            data = pd.read_json('https://raw.githubusercontent.com/nlp-compromise/penn-treebank/f96fffb8e78a9cc924240c27b25fb1dcd8974ebf/penn-data.json')
            sentences = data[0].tolist()
            self.sentences = sentences[:num_sentences]

        def __len__(self):
            """Returns the number of samples in the dataset."""
            return len(self.sentences)

        def __getitem__(self, index):
            """returns a sentence"""
            sentence= self.sentences[index]
            return {'x':sentence, 'label':None}

    dataset = PENNTREEBANK()
    batch_size = hyperparams['BATCH_SIZE']
    train_size = int(0.7 * len(dataset))  # 70% for training
    val_size = int(0.1 * len(dataset)) # 10% for validation
    test_size = len(dataset) - train_size - val_size # 20% for testing

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

