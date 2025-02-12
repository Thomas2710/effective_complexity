from effective_complexity.model import MLP
from effective_complexity.functions import compute_distrib
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms, utils
from effective_complexity.datasets import collate_fn
import torch
import numpy as np
from tqdm import tqdm
import math


def dataset_synthetic(hyperparams):
    class SYNTHETIC(Dataset):

        def __init__(self):
            """
            Custom PyTorch Dataset that stores an array of dictionaries.

            Args:
                data (list of dicts): Each dictionary represents a data point with keys as feature names.
            """
            self.data = []

        def __len__(self):
            """Returns the number of samples in the dataset."""
            return len(self.data)

        def __getitem__(self, idx):
            """
            Retrieves a single sample from the dataset.

            Args:
                idx (int): Index of the sample.

            Returns:
                dict: A dictionary containing the features of the indexed sample.
            """
            return self.data[idx]
        
        def add_item(self, item):
            """
            Adds a new item (dictionary) to the dataset.

            Args:
                item (dict): A dictionary containing the new data sample.
            """
            if not isinstance(item, dict):
                raise ValueError("Item must be a dictionary.")
            self.data.append(item)
    
    train_percent=0.7
    test_percent = 0.2
    val_percent = 0.1

    # Number of points to sample
    num_samples = hyperparams['num_samples']
    DIMS = hyperparams['DIMS']
    COV = hyperparams['COV']
    MU = hyperparams['MU']
    BATCH_SIZE = hyperparams['BATCH_SIZE']
    
    train_dataset = SYNTHETIC()
    test_dataset = SYNTHETIC()
    val_dataset = SYNTHETIC()

    with torch.no_grad():
        #Generate data and W
        mu_sample = np.full(DIMS,MU)
        covariance_sample = np.diag(np.full(DIMS,COV))

        # Sample points
        samples = np.random.multivariate_normal(mu_sample, covariance_sample, num_samples)
        samples = torch.from_numpy(samples).float()
        #Define orthogonal vectors
        w1 = torch.tensor([1,2,2])
        w2 = torch.tensor([-2,1,0])
        w3 = torch.tensor([-2,-2,1])
        W = torch.stack([w1,w2,w3], dim = 0).t().float()


        #generate f_x
        # Instantiate the MLP (f(x))
        input_size = 3  # Number of input features
        hidden_sizes = [64, 32, 64]  # Hidden layer sizes
        output_size = 3  # Number of output features (e.g., classification classes)
        mlp_model = MLP(input_size, hidden_sizes, output_size)
        #mlp_model.apply(initialize_weights)
        f_x = torch.stack([mlp_model(x) for x in samples], dim=0)
    

        #generate y
        print('Computing ground truth labels')
        for x in tqdm(f_x):
            if len(train_dataset) < math.floor(num_samples*train_percent):
                train_dataset.add_item({'x':x, 'label':compute_distrib(x,W)})
            elif len(val_dataset) < math.floor(num_samples*val_percent):
                val_dataset.add_item({'x':x, 'label':compute_distrib(x,W)})
            else:
                test_dataset.add_item({'x':x, 'label':compute_distrib(x,W)})

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size = 128, shuffle = True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader