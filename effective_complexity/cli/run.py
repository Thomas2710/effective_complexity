# Copyright 2023 Janek Bevendorff
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

def run(model, dataset):
    #Run checks and load right configuration
    # Load hyperparameters from YAML file
    with open(os.path.join(os.getcwd(), 'configs', 'model',''+model+'.yaml'), 'r') as file:
        model_hyperparams = yaml.safe_load(file)
    with open(os.path.join(os.getcwd(), 'configs', 'dataset',''+dataset+'.yaml'), 'r') as file:
        dataset_hyperparams = yaml.safe_load(file)
    with open(os.path.join(os.getcwd(), 'configs', 'general.yaml'), 'r') as file:
        general_hyperparams = yaml.safe_load(file)

    

    #Load model
    from effective_complexity.models import get_model_class
    model_class = get_model_class(model)
    model = model_class[0](model_hyperparams)

    #Load dataset
    from effective_complexity.datasets import get_dataset_class
    dataset_class = get_dataset_class(dataset)
    dataloader = dataset_class[0](dataset_hyperparams)


    hyperparams = (general_hyperparams, model_hyperparams, dataset_hyperparams)
    #Run main function
    from effective_complexity.main import identify
    identify(dataloader, model, hyperparams)
 