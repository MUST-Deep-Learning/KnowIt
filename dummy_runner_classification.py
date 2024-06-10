# Users interact with KnowIt by either constructing an experiment script such as this one or importing KnowIt into their
# own scripts and using specific functionalities in KnowIt.

# This script shows an example of how to perform time series classification.
# See dummy_runner_regression.py for a version that performs time series regression and contains more details.

# ----------------------------------------------------------------------------------------------------------------------
# IMPORTING THE KNOWIT MODULE
# ----------------------------------------------------------------------------------------------------------------------

from knowit import KnowIt

# ----------------------------------------------------------------------------------------------------------------------
# CONSTRUCTING AN INSTANCE OF KNOWIT
# ----------------------------------------------------------------------------------------------------------------------

# KI = KnowIt(custom_exp_dir='/path/to/my/new/experiment/directory')
KI = KnowIt()

# ----------------------------------------------------------------------------------------------------------------------
# TRAINING A MODEL
# ----------------------------------------------------------------------------------------------------------------------

model_name = "my_new_penguin_model"
data_args = {'name': 'penguin_42_debug',
             'task': 'classification',
             'in_components': ['accX', 'accY', 'accZ'],
             'out_components': ['PCE'],
             'in_chunk': [-25, 25],
             'out_chunk': [0, 0],
             'split_portions': [0.6, 0.2, 0.2],
             'batch_size': 32,
             'split_method': 'chronological',
             'scaling_tag': 'in_only',
             'min_slice': 100}
arch_args = {'task': 'classification',
             'name': 'CNN'}
trainer_args = {'loss_fn': 'weighted_cross_entropy',
                'optim': 'Adam',
                'max_epochs': 5,
                'learning_rate': 0.001,
                'learning_rate_scheduler': {'ExponentialLR': {'gamma': 0.9}},
                'task': 'classification'}
KI.train_model(model_name=model_name, kwargs={'data': data_args, 'arch': arch_args, 'trainer': trainer_args})

# ----------------------------------------------------------------------------------------------------------------------
# GENERATE MODEL PREDICTIONS
# ----------------------------------------------------------------------------------------------------------------------

KI.generate_predictions(model_name=model_name, kwargs={'predictor': {'prediction_set': 'eval'}})

# ----------------------------------------------------------------------------------------------------------------------
# INTERPRET MODEL PREDICTIONS
# ----------------------------------------------------------------------------------------------------------------------

interpret_args = {'interpretation_method': 'DeepLift',
                  'interpretation_set': 'eval',
                  'selection': 'success',
                  'size': 100}

KI.interpret_model(model_name=model_name, kwargs={'interpreter': interpret_args})


exit(101)
