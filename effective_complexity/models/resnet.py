import torch
import torch.nn as nn
from effective_complexity.models.models import DistributionComponents
import torch.nn.functional as F
import torchvision.models as models

def model_resnet(hyperparams):
    class RESNET(nn.Module, DistributionComponents):
        def __init__(self, embedding_size=3,output_size=3, flatten_input=False):
            super(RESNET, self).__init__()
            # Define layers
            self.resnet50 = models.resnet50(weights=None) 
            self.fx = nn.Linear(self.resnet50.fc.weight.shape[0], embedding_size)
            self.W = nn.Linear(embedding_size, output_size)

        def forward(self, x):
            x = self.resnet50(x)
            x = self.fx(x)
            return x
        
        def get_fx(self, x):
            return x
        
        #Probably needs to change
        def get_unembeddings(self, y):
            return torch.matmul(self.W.weight.t(),y)
        
        def get_W(self):
            return self.W.weight


    embedding_size = hyperparams['embedding_size']
    num_classes = hyperparams['num_classes']
    model = RESNET(embedding_size, num_classes)
    return model