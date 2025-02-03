import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def model_mlp(hyperparams):
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
        
    input_size = hyperparams['input_size']
    hidden_sizes = hyperparams['hidden_sizes']
    output_size = hyperparams['output_size']
    model = MLP(input_size, hidden_sizes, output_size)
    return model
    

def list_models():
    models = [n.replace('model_', '')  for n, m in globals().items()
            if n.startswith('model_')]
    return models


def get_model_class(model_name):
    model= [m  for n, m in globals().items()
        if n.startswith('model_'+model_name)]

    return model