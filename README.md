# KnowIt
KnowIt (**Know**ledge discovery **I**n **T**ime series data) is a toolkit to train and interpret deep time series models.\
It is developed in Python and uses [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) for model building and [Captum](https://github.com/pytorch/captum) for model interpreting.

## Installation

 - Clone the KnowIt directory locally.
 - Create a conda environment with the provided ``KnowIt/environment.yml`` file.

## Usage

See the ``KnowIt/dummy_runner_regression.py`` and ``KnowIt/dummy_runner_classification.py`` scripts for full examples on usage.
A brief summary follows:
```python
# Import the KnowIt module
from knowit import KnowIt

# Instantiate an instance of the KnowIt module
KI = KnowIt()

# Import a new dataset
KI.import_dataset({'data_import_args': import_args})

# Train a model
KI.train_model(model_name='new_model_name', 
               args={'data': data_args, 
                     'arch': arch_args, 
                     'trainer': trainer_args})

# Generate model predictions
KI.generate_predictions(model_name='new_model_name', 
                        args={'predictor': predictor_args})

# Interpret model predictions
KI.interpret_model(model_name='new_model_name', 
                   args={'interpreter': interpret_args})
```

## Features

KnowIt currently supports the following tasks on uni- or multivariate equidistantly spaced time series data:
 - Time series regression (uni- or multivariate, single- or multistep, autoregressive or not)
 - Time series classification (multivariate, single-step, binary or multi-class)
 - Time series forcasting (by means of autoregressive time series regression with a delay)

KnowIt provides convenient entry points to import custom architectures (PyTorch based) 
and custom dataset (Pandas based).

## Contact

 - Main author: tiantheunissen@gmail.com
 - Main author: randlerabe@gmail.com

## Coming soon

 - Package installation options
 - Live API reference docs
 - Tutorials
 - Citation link

