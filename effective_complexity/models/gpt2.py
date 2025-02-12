import torch
import torch.nn as nn
from effective_complexity.models.models import DistributionComponents
from transformers import *

def model_gpt(hyperparams):
    class GPT(nn.Module, DistributionComponents):
        def __init__(self, model_id, output_size, flatten):
            super(GPT, self).__init__()
            models = {
                'gpt2': (GPT2Model, GPT2Tokenizer, 'gpt2'),
                'gpt2-medium': (GPT2Model, GPT2Tokenizer, 'gpt2-medium'),
                'gpt2-large': (GPT2Model, GPT2Tokenizer, 'gpt2-large'),
                'gpt2-xl': (GPT2Model, GPT2Tokenizer, 'gpt2-xl'),
            }

            model_class, tokenizer_class, pretrained_weights = models[model_id]
            self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
            #model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True, cache_dir=cache_dir)
            self.model = model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
            self.map = nn.Linear(768, output_size)
            self.flatten = flatten

        def text2features(self, text):
            input_ids = torch.tensor([self.tokenizer.encode(text, add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
            with torch.no_grad():
                hidden_states = self.model(input_ids=input_ids)
            return hidden_states

            
        def forward(self, x):
            if self.flatten:
                x = x.reshape(x.shape[0], 32 * 32 * 3)
            #print(x.shape, x.min(), x.max(), x)
            x = torch.multiply(x,100).long()
            print(x.min(), x.max())
            hiddens = self.model(x).last_hidden_state
            return hiddens
        
        def get_fx(self, x):
            return x.mean(dim=1)
        
        def get_unembeddings(self, y):
            return torch.stack([self.map(i) for i in y])
        
        def get_W(self):
            return self.map.weight
        
    output_size = hyperparams['output_size']
    model_id = hyperparams['model_id']
    flatten = hyperparams['flatten_input']
    model = GPT(model_id, output_size, flatten)
    return model