import torch
import torch.nn as nn
from effective_complexity.models.models import DistributionComponents
import torch.nn.functional as F

def model_cnn(hyperparams):
    class CNN(nn.Module, DistributionComponents):
        def __init__(self, input_size,num_classes=10):
            super(CNN, self).__init__()

            # Convolutional layers
            self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=16, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

            # Pooling layer
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            # Fully connected layers
            self.fc1 = nn.Linear(64 * 4 * 4, 128)  # Adjust based on input size
            #self.fc2 = nn.Linear(128, num_classes)
            self.W = nn.Linear(128, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))

            x = torch.flatten(x, start_dim=1)  # Flatten for FC layers

            x = F.relu(self.fc1(x))
            #x = self.fc2(x)  # No activation (applied in loss function)

            return x
        
        def get_fx(self, x):
            return x
        
        def get_unembeddings(self, y):
            print('in unembeddins, shape of W is ', self.W.weight.shape, 'shape of y is', y.shape)
            return torch.matmul(self.W.weight.t(),y)
        
        def get_W(self):
            return self.W.weight
        
    flatten_input = hyperparams['flatten_input']
    input_size = hyperparams['input_size']
    output_size = hyperparams['output_size']
    model = CNN(input_size, num_classes=output_size)
    return model