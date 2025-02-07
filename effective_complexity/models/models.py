import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# Install and import Huggingface Transformer models
#!pip install transformers ftfy spacy
from transformers import *
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
num_layers = 13
import numpy as np



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

def model_gpt(hyperparams):
    class GPT(nn.Module):
        def __init__(self, model_id ,input_size, hidden_sizes, output_size):
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

        def text2features(self, text):
            input_ids = torch.tensor([self.tokenizer.encode(text, add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
            with torch.no_grad():
                hidden_states = self.model(input_ids=input_ids)
            return hidden_states


        '''
        # Compute model activations of data for an architecture.
        # Returns a 2D numpy array with [n_datapoints, n_features].
        # Each word is one datapoint, and converts to one feature vector.
        def get_activations(self, f_model, data, num_layers=num_layers):
                _data = data[i]
                hiddens = f_model(_data)[2] # Get all hidden layers
                hiddens = [h.numpy() for h in hiddens]
                hiddens = np.concatenate(hiddens, axis=0)
                hiddens = hiddens[-num_layers:]
                # hiddens.shape = (num_layers, num_datapoints, num_features)
                h.append(hiddens)
            h = np.concatenate(h, axis=1)
            print('Activations shape: ', h.shape)
            return h
        '''

            
        def forward(self, x):
            hiddens = self.model(x)[2]
            print('hidden_shape is ', hiddens.shape)
            return hiddens
        
    input_size = hyperparams['input_size']
    hidden_sizes = hyperparams['hidden_sizes']
    output_size = hyperparams['output_size']
    model_id = hyperparams['model_id']
    print('MODEL ID', model_id)
    model = GPT(model_id, input_size, hidden_sizes, output_size)
    return model
  


def list_models():
    models = [n.replace('model_', '')  for n, m in globals().items()
            if n.startswith('model_')]
    return models


def get_model_class(model_name):
    model= [m  for n, m in globals().items()
        if n.startswith('model_'+model_name)]

    return model
