from abc import ABC, abstractmethod

class DistributionComponents(ABC):
    @abstractmethod
    def get_unembeddings(self, y):
        pass

    @abstractmethod
    def get_W(self):
        pass

    @abstractmethod
    def get_fx(self, x):
        pass

def model_gpt(hyperparams):
    from effective_complexity.models.gpt2 import model_gpt as mg
    return mg(hyperparams)

def model_mlp(hyperparams):
    from effective_complexity.models.mlp import model_mlp as mm
    return mm(hyperparams)

def list_models():
    models = [n.replace('model_', '')  for n, m in globals().items()
            if n.startswith('model_')]
    return models


def get_model_class(model_name):
    model= [m  for n, m in globals().items()
        if n.startswith('model_'+model_name)]

    return model
