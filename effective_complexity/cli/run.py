import os
import logging
import click
import yaml
from effective_complexity.globals import *

#os.environ["QT_QPA_PLATFORM"] = "wayland"

@click.command()
@click.option('-m', '--model', type=click.Choice([*MODELS_ALL]), default='mlp',
              help='Models that satisfy paper property', multiple=False)
@click.option('-d', '--dataset', type=click.Choice([*DATASETS_ALL]), default='synthetic',
              help='Datasets', multiple=False)
@click.option('-t', '--train', is_flag=True,
              help='Train model')
@click.option('-v', '--verbose', is_flag=True,
              help='Enables verbose mode')

def run(model, dataset, train, verbose):
    #Run checks and load right configuration
    # Load hyperparameters from YAML file
    with open(os.path.join(os.getcwd(), 'configs', 'model',''+model+'.yaml'), 'r') as file:
        model_hyperparams = yaml.safe_load(file)
    with open(os.path.join(os.getcwd(), 'configs', 'dataset',''+dataset+'.yaml'), 'r') as file:
        dataset_hyperparams = yaml.safe_load(file)
    with open(os.path.join(os.getcwd(), 'configs', 'general.yaml'), 'r') as file:
        general_hyperparams = yaml.safe_load(file)

    if not model_hyperparams:
        model_hyperparams = dict()
    if not dataset_hyperparams:
        dataset_hyperparams = dict()
    if not general_hyperparams:
        general_hyperparams = dict()

        
    #Load dataset
    from effective_complexity.datasets import get_dataset_class
    dataset_class = get_dataset_class(dataset)
    dataset_hyperparams['embedding_size'] = general_hyperparams['embedding_size']
    if dataset == 'synthetic' and (model == 'resnet' or model == 'cnn'):
        dataset_hyperparams['DIMS'] = [3,32,32]
        dataset_hyperparams['hidden_sizes'] = [1024, 512, 10]
        dataset_hyperparams['flatten_input'] = True
        dataset_hyperparams['input_size'] = 32*32*3
    dataloader = dataset_class[0](dataset_hyperparams)

    #Get dataset number of classes
    batch_sample = next(iter(dataloader[0]))
    num_classes = batch_sample['label'].shape[1]

    #Load model
    from effective_complexity.models import get_model_class
    model_class = get_model_class(model)
    model_hyperparams['embedding_size'] = general_hyperparams['embedding_size']

    model_hyperparams['num_classes'] = num_classes
    model = model_class[0](model_hyperparams)


    hyperparams = (general_hyperparams, model_hyperparams, dataset_hyperparams)
    #Run main function
    from effective_complexity.main import identify
    identify(dataloader, model, hyperparams, train)
 