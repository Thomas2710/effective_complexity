from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import CIFAR10 as CIFAR10
from torchvision.datasets import CIFAR100
import torch.nn.functional as F
import torch
from tqdm import tqdm
import math
import numpy as np
from effective_complexity.model import MLP
from effective_complexity.functions import compute_distrib
import os

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

    return train_dataset, val_dataset, test_dataset
    


def dataset_cifar(hyperparams):
    class CIFAR(CIFAR10):
        def __init__(self, root=os.path.join(os.getcwd(),"datasets","data"), train=True, transform=None, target_transform=None, download=True):
            super().__init__(root=root, train=train, transform=transform, target_transform=target_transform, download=download)


        def __len__(self):
            """Returns the number of samples in the dataset."""
            return len(self.data)

        def __getitem__(self, index):
            """Override the default method to modify how images and labels are returned"""
            image, label = super().__getitem__(index)  # Get image and label using parent class method

            # Convert label to one-hot encoding
            label_one_hot = F.one_hot(torch.tensor(label), num_classes=10).float()

            return image, label_one_hot  # Return image with modified label

    # Define transformations (normalization & augmentation)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = CIFAR(train=True, transform=transform)
    test_dataset = CIFAR(train=False, transform=transform)

    # Define sizes for train & validation splits
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation

    # Randomly split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    #each dataset is a list(10k items) of tuples (x,label)) where x and label are tensors
    return train_dataset, val_dataset, test_dataset



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