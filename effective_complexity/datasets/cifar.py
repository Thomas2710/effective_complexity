from torchvision.datasets import CIFAR10 as CIFAR10
from torchvision.datasets import CIFAR100
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms, utils
from effective_complexity.datasets import collate_fn
import torch
import os 
import torch.nn.functional as F

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

            return {'x':image, 'label':label_one_hot}  # Return image with modified label

    BATCH_SIZE = hyperparams['BATCH_SIZE']

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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2, collate_fn=collate_fn)
   
    return train_loader, val_loader, test_loader