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


@click.command()
@click.option('-m', '--model', type=click.Choice([*MODELS_ALL]), default=['mlp'],
              help='Models that satisfy paper property', multiple=False)
def run(model):
    #Run checks and load right configuration
    # Load hyperparameters from YAML file
    with open(os.path.join(os.getcwd(), 'configs', ''+model+'.yaml'), 'r') as file:
        hyperparams = yaml.safe_load(file)


    #Load model
    from effective_complexity.models import get_model_class
    from effective_complexity.main import identify
    model_class = get_model_class(model)
    #print(type(model_class[1]), model_class[1])
    model = model_class[0](hyperparams)
    identify(model, hyperparams)
 