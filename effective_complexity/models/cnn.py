import torch
import torch.nn as nn
from effective_complexity.models.models import DistributionComponents
import torch.nn.functional as F

def model_cnn(hyperparams):
    class CNN(nn.Module, DistributionComponents):
        def __init__(self, input_size,embedding_size=10, output_size=3):
            super(CNN, self).__init__()

            # Convolutional layers
            self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=16, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

            # Pooling layer
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            # Fully connected layers
            self.fc1 = nn.Linear(64 * 4 * 4, 128)  # Adjust based on input size
            self.fx = nn.Linear(128, embedding_size)
            self.W = nn.Linear(embedding_size, output_size)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))

            x = torch.flatten(x, start_dim=1)  # Flatten for FC layers

            x = F.relu(self.fc1(x))
            output = self.fx(x)  # No activation (extracting embeddings)

            return output
        
        def get_fx(self, x):
            return x
        
        def get_unembeddings(self, y):
            return torch.matmul(self.W.weight.t(),y)
        
        def get_W(self):
            return self.W.weight
    
    embedding_size = hyperparams['embedding_size']
    flatten_input = hyperparams['flatten_input']
    input_size = hyperparams['input_size']
    num_classes = hyperparams['num_classes']
    model = CNN(input_size, embedding_size=embedding_size, output_size=num_classes)
    return model