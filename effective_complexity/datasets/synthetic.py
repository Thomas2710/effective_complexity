from effective_complexity.models import get_model_class
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms, utils
from effective_complexity.datasets import collate_fn
import torch
import numpy as np
from tqdm import tqdm
import math

eps = 10e-4

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
        
        def add_items(self, items):
            """
            Adds tensor as dataset.

            Args:
                items (dict): A dict containing data samples as tensors.
            """

            if not isinstance(items, dict):
                raise ValueError("Item must be a dict.")
            self.data= [dict(zip(items.keys(), values)) for values in zip(*items.values())]
    
    train_percent=0.7
    test_percent = 0.2
    val_percent = 0.1

    # Number of points to sample
    num_samples = hyperparams['num_samples']
    DIMS = hyperparams['DIMS']
    COV = hyperparams['COV']
    MU = hyperparams['MU']
    BATCH_SIZE = hyperparams['BATCH_SIZE']
    num_classes = hyperparams['num_classes']
    
    one_hots = torch.eye(num_classes).float()

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
        w1 = torch.tensor([1,1,1])
        w2 = torch.tensor([1,-1,0])
        w3 = torch.tensor([1,1,-2])
        W = torch.stack([w1,w2,w3], dim = 0).t().float()


        #generate f_x
        # Instantiate the MLP (f(x))
        model_class = get_model_class('mlp')
        hyperparams['embedding_size'] = 3
        mlp_model = model_class[0](hyperparams)
        #mlp_model.apply(initialize_weights)
        


        embedding_outputs = mlp_model(samples)
        f_x = mlp_model.get_fx(embedding_outputs)
        unembedding = torch.matmul(W, one_hots)
        logits = torch.matmul(f_x, unembedding)

        # Define the softmax layer
        softmax = torch.nn.Softmax(dim=-1)
        distrib = softmax(logits)

        sum = torch.sum(distrib, dim=1)
        assert torch.all(sum > 1 - eps) and torch.all(sum < 1 + eps)

        #Shuffle sample tensor
        #torch.randperm?

        #Add to the dataset
        train_end_index = math.floor(num_samples*train_percent)
        train_dataset.add_items({'x':f_x[:train_end_index, :] , 'label':distrib[:train_end_index, :]})
        val_start_index = train_end_index
        val_end_index = math.floor(num_samples*(train_percent+val_percent))
        val_dataset.add_items({'x':f_x[val_start_index:val_end_index, :], 'label': distrib[val_start_index:val_end_index, :]})
        test_start_index = val_end_index
        test_dataset.add_items({'x': f_x[test_start_index:, :], 'label':distrib[test_start_index:, :]})


    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size = 128, shuffle = True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader