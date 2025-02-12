import torch
import torch.nn as nn
from effective_complexity.models.models import DistributionComponents
import torch.nn.functional as F

def model_mlp(hyperparams):
    class MLP(nn.Module, DistributionComponents):
        def __init__(self, input_size, hidden_sizes, output_size, flatten_input=False):
            super(MLP, self).__init__()
            # Define layers
            self.flatten = flatten_input
            self.layers = nn.ModuleList()
            in_features = input_size
            for hidden_size in hidden_sizes:
                self.layers.append(nn.Linear(in_features, hidden_size))
                in_features = hidden_size
            self.output_layer = nn.Linear(in_features, output_size)

        def forward(self, x):
            if self.flatten:
                x = x.reshape(x.shape[0], 32 * 32 * 3)

            # Pass input through each layer with ReLU activation
            for layer in self.layers:
                x = F.relu(layer(x))
            # Output layer
            #x = self.output_layer(x)
            return x
        
        def get_fx(self, x):
            return x
        
        def get_unembeddings(self, y):
            return torch.stack([self.output_layer(i) for i in y])
        
        def get_W(self):
            return self.output_layer.weight
        
    flatten_input = hyperparams['flatten_input']
    input_size = hyperparams['input_size']
    hidden_sizes = hyperparams['hidden_sizes']
    output_size = hyperparams['output_size']
    model = MLP(input_size, hidden_sizes, output_size, flatten_input)
    return model