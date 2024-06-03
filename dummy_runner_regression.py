# Users interact with KnowIt by either constructing an experiment script such as this one or importing KnowIt into their
# own scripts and using specific functionalities in KnowIt.

# ----------------------------------------------------------------------------------------------------------------------
# IMPORTING THE KNOWIT MODULE
# ----------------------------------------------------------------------------------------------------------------------

from knowit import KnowIt

# ----------------------------------------------------------------------------------------------------------------------
# CONSTRUCTING AN INSTANCE OF KNOWIT
# ----------------------------------------------------------------------------------------------------------------------

# You can construct an instance of the KnowIt module as indicated in the examples below:

# Providing no additional arguments means that the experiment output directory will be temporary.
# This means that the entire directory, along with all outputs, will be deleted when the instance
# of KnowIt is removed by the garbage collector. This is useful for debugging or when KnowIt is
# used as a submodule of a larger codebase.

# KI = KnowIt()

# Providing a custom_exp_dir path means that the constructed instance of KnowIt will be
# associated with a static (potentially pre-existing) experiment output directory.
# This is useful for long-term experimentation.

KI = KnowIt(custom_exp_dir='/path/to/my/new/experiment/directory')

# ----------------------------------------------------------------------------------------------------------------------
# IMPORTING A CUSTOM ARCHITECTURE
# ----------------------------------------------------------------------------------------------------------------------

# KnowIt comes with a set of default architectures (see default_archs directory).
# However, if you want to import a custom one you can call the following function.
# The given script will be checked and saved in the experiment output directory.
# See the default_archs.arch_how_to.md file along with the default architectures
# for more detail.

# KI.import_arch(new_arch_path='/path/to/my/custom/arch.py')

# ----------------------------------------------------------------------------------------------------------------------
# IMPORTING A DATASET
# ----------------------------------------------------------------------------------------------------------------------

# KnowIt has two default datasets for debugging (see default_datasets directory).
# In order to handle external data you can import it into your experiment output directory as follows.
# See the default_datasets.dataset_how_to.md file for more details.

# import_args = {'path': '/path/to/my/custom/data.pickle',
#                'base_nan_filler': 'linear',
#                'nan_filled_components': ['x1', 'x2', 'x3', 'x4', 'y1']}
# KI.import_dataset({'data_import_args': import_args})

# ----------------------------------------------------------------------------------------------------------------------
# CHECK AVAILABLE DATASETS & ARCHITECTURES
# ----------------------------------------------------------------------------------------------------------------------

# check the state of global arguments for current instance of KnowIt
print('CURRENT GLOBAL ARGUMENTS:')
print(KI.global_args())
# check the available datasets for current instance of KnowIt
print('CURRENT AVAILABLE DATASETS:')
print(KI.available_datasets())
# check the available architectures for current instance of KnowIt
print('CURRENT GLOBAL ARCHITECTURES:')
print(KI.available_archs())
# check the metadata of a particular dataset for current instance of KnowIt
print('SUMMARY OF SOME DATASET:')
print(KI.summarize_dataset('synth_1'))

# ----------------------------------------------------------------------------------------------------------------------
# TRAINING A MODEL
# ----------------------------------------------------------------------------------------------------------------------

# To train a model you need to provide a model name and a set of arguments.
# - 'data' arguments relate to how the data should be split, scaled, sampled, and processed by the model.
# - 'arch' arguments relate to what model architecture should be trained to perform what task.
# - 'trainer' arguments relate to what methods should be used to train the model.
# see the setup.setup_action_args.py scripts for some options and default values.

model_name = "my_new_model"
data_args = {'name': 'synth_1',
             'task': 'regression',
             'in_components': ['x1', 'x2', 'x3', 'x4'],
             'out_components': ['y1'],
             'in_chunk': [-5, 5],
             'out_chunk': [0, 0],
             'split_portions': [0.6, 0.2, 0.2],
             'batch_size': 64,
             'split_method': 'instance-random',
             'scaling_tag': 'full'}
arch_args = {'task': 'regression',
             'name': 'MLP',
             'arch_hps': {'dropout': 0.0,
                          'width': 512}}
trainer_args = {'loss_fn': 'mse_loss',
                'optim': 'Adam',
                'max_epochs': 10,
                'learning_rate': 0.01,
                'task': 'regression'
                }

KI.train_model(model_name=model_name, args={'data': data_args, 'arch': arch_args, 'trainer': trainer_args})


# ----------------------------------------------------------------------------------------------------------------------
# GENERATE MODEL PREDICTIONS
# ----------------------------------------------------------------------------------------------------------------------

# To generate predictions for a trained model you need to provide a model name and a set of arguments.
# - 'predictor' arguments relate to what prediction points to predict on.

# KI.generate_predictions(model_name=model_name, args={'predictor': {'prediction_set': 'train'}})
# KI.generate_predictions(model_name=model_name, args={'predictor': {'prediction_set': 'valid'}})
KI.generate_predictions(model_name=model_name, args={'predictor': {'prediction_set': 'eval'}})

# ----------------------------------------------------------------------------------------------------------------------
# INTERPRET MODEL PREDICTIONS
# ----------------------------------------------------------------------------------------------------------------------

# To interpret predictions of a trained model you need to provide a model name and a set of arguments.
# - 'interpreter' arguments relate to what prediction points to interpret on.
# NOTE: In order to visualize feature attributions, the corresponding predictions must have been generated.

interpret_args = {'interpretation_method': 'DeepLift',
                  'interpretation_set': 'eval',
                  'selection': 'random',
                  'size': 100}
KI.interpret_model(model_name=model_name, args={'interpreter': interpret_args})

exit(101)
