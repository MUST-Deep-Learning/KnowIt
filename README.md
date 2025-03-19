# Knowledge discovery in time series data

![KI_logo.png](KI_logo.png)

---

KnowIt (**Know**ledge discovery **I**n **T**ime series data) is a toolkit to build and interpret deep time series models. 
It is developed in Python and uses [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) for model building and [Captum](https://github.com/pytorch/captum) for model interpreting.

## Installation

 - Clone the KnowIt directory locally
 - Create a [conda environment](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) with the provided ``KnowIt/environment.yml`` file
 - Installation package TBD

## Usage overview

```python
# Import the KnowIt module
from knowit import KnowIt

# Define an experiment output directory
exp_out_dir = 'output_path'

# Instantiate a KnowIt object
KI = KnowIt(exp_out_dir)

# Import a new dataset from a pickled pandas DataFrame
KI.import_dataset({'data_import': {'path': 'dummy_df.pkl'}})

# Define your dataset
data_args = {'name': 'dummy_dataset',
             'task': 'regression',
             'in_components': ['x'],
             'out_components': ['y'],
             'in_chunk': [-10, 0],
             'out_chunk': [0, 0],
             'split_portions': [0.6, 0.2, 0.2],
             'batch_size': 64}

# Define your architecture
arch_args = {'task': 'regression',
             'name': 'TCN'}

# Define your trainer
trainer_args = {'loss_fn': 'mse_loss',
                'optim': 'Adam',
                'max_epochs': 10,
                'learning_rate': 0.01,
                'task': 'regression'}

# Name your model
model_name = 'new_model_name'

# Train your model
KI.train_model(model_name=model_name, 
               kwargs={'data': data_args, 
                       'arch': arch_args, 
                       'trainer': trainer_args})

# Generate model predictions
KI.generate_predictions(model_name=model_name, 
                        kwargs={'predictor': {'prediction_set': 'valid'}})

# Interpret model predictions
KI.interpret_model(model_name=model_name, 
                   kwargs={'interpreter': 
                               {'interpretation_method': 'DeepLift', 
                                'interpretation_set': 'valid', 
                                'selection': 'random', 
                                'size': 100}})
```

## Current Features
 - **Flexible problem class definition** allows facilitating various time series modeling 
tasks, including: uni- and multivariate, single- and multistep, time series regression, classification, forecasting, and detection. 
 - A growing list of default **deep time series architectures**. Currently includes: MLP, 
CNN, TCN, and LSTM.
 - Convenient **entry points to import** custom architectures (PyTorch based) and custom dataset (Pandas based).
 - **Automated data management** (storing, splitting, scaling, sampling, loading, padding etc.) of complex 
time series data.
 - Convenient **feature attribution** extraction and storage.
 - **Hyperparameter tuning** through Pytorch Lightning and Weights and Biases API.
 - **Extendable design** for additional future training and interpretability paradigms.

## Current Limitations
 - Data is assumed to be equidistantly sampled and time-indexed time series data. 
Gaps are removed during preprocessing to construct contiguous blocks of data for model training.
 - Data is stored on and loaded from disk in contiguous slices. These slices have to fit into memory.
 - Currently only support "stateless" model training. This means that temporal dependencies 
are assumed to not extend beyond the model's input chunk. In other words, information related to 
the model prediction is not propagated across batches.

## Coming soon

 - Adding Transformer-based models to default architectures
 - Stateful model training
 - Package installation options
 - Testbench for synthetic data experimentation
 - Additional feature attribution visualizations

We are open to any suggestions.

## Citation
If you make use of KnowIt, we kindly request that you cite it with the provided citation file.
This link will be updated to an official link soon.

## Acknowledgments

This project is made possible due to funding and support from:
- MUST Deep Learning
- North-West University (NWU), South Africa
- National Institute for Theoretical & Computational Sciences (NITheCS), South Africa
- Centre for Artificial Intelligence Research (CAIR), South Africa

Contributors:
- **Marthinus Wilhelmus Theunissen** (tiantheunissen@gmail.com)
- **Randle Rabe** (randlerabe@gmail.com)

