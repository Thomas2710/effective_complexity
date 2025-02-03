import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        # Define layers
        self.layers = nn.ModuleList()
        in_features = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(in_features, hidden_size))
            in_features = hidden_size
        self.output_layer = nn.Linear(in_features, output_size)

    def forward(self, x):
        # Pass input through each layer with ReLU activation
        for layer in self.layers:
            x = F.relu(layer(x))
        # Output layer
        x = self.output_layer(x)
        return x


# Custom weight initialization function
def initialize_weights(module):
    if isinstance(module, nn.Linear):
        # Initialize weights randomly (e.g., with normal distribution)
        init.normal_(module.weight, mean=0.0, std=1)  # Random normal initialization
        if module.bias is not None:
            init.constant_(module.bias, 0)  # Initialize biases to 0
