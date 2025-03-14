import torch
import torch.nn as nn
from effective_complexity.models.models import DistributionComponents
import torch.nn.functional as F

def model_lstm(hyperparams):
    class LSTM(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=3):
            super(LSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.W = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            x = x.unsqueeze(dim=1)
            out, _ = self.lstm(x) 
            #out = self.W(out[:, -1, :])
            # Take the last hidden state and pass to classifier
            return out

        def get_fx(self, x):
            return x.squeeze(dim=1)
        
        def get_unembeddings(self, y):
            return torch.matmul(self.W.weight.t(),y)
        
        def get_W(self):
            return self.W.weight
    
    embedding_size = hyperparams['embedding_size']
    input_size = hyperparams['input_size']
    num_classes = hyperparams['num_classes']
    model = LSTM(input_size, hidden_size=embedding_size, output_size=num_classes)
    return model