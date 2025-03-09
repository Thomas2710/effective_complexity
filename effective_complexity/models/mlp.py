import torch
import torch.nn as nn
from effective_complexity.models.models import DistributionComponents
import torch.nn.functional as F

def model_mlp(hyperparams):
    class MLP(nn.Module, DistributionComponents):
        def __init__(self, input_size, hidden_sizes, embedding_size=3,output_size=3, flatten_input=False):
            super(MLP, self).__init__()
            # Define layers
            self.flatten = flatten_input
            self.layers = nn.ModuleList()
            in_features = input_size
            for hidden_size in hidden_sizes:
                self.layers.append(nn.Linear(in_features, hidden_size))
                in_features = hidden_size
            self.fx = nn.Linear(in_features, embedding_size)
            self.W = nn.Linear(embedding_size, output_size)

        def forward(self, x):
            if self.flatten:
                x = x.reshape(x.shape[0], 32 * 32 * 3)
            
            # Pass input through each layer with ReLU activation
            for layer in self.layers:
                x = F.relu(layer(x))
            # Output layer
            x = self.fx(x)
            return x
        
        def get_fx(self, x):
            return x
        
        #Probably needs to change
        def get_unembeddings(self, y):
            return torch.matmul(self.W.weight.t(),y)
        
        def get_W(self):
            return self.W.weight

    flatten_input = hyperparams['flatten_input']
    input_size = hyperparams['input_size']
    hidden_sizes = hyperparams['hidden_sizes']
    embedding_size = hyperparams['embedding_size']
    num_classes = hyperparams['num_classes']
    model = MLP(input_size, hidden_sizes,embedding_size, num_classes, flatten_input)
    return model