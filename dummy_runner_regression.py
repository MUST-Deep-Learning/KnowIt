# ----------------------------------------------------------------------------------------------------------------------
# IMPORTING THE KNOWIT MODULE
# ----------------------------------------------------------------------------------------------------------------------

from knowit import KnowIt

# ----------------------------------------------------------------------------------------------------------------------
# CONSTRUCTING AN INSTANCE OF KNOWIT
# ----------------------------------------------------------------------------------------------------------------------

# You can construct an instance of the KnowIt module as indicated in the examples below:

# Providing no additional arguments means that the experiment output directory will be temporary.
# This means that the entire directory will be deleted after the instance of KnowIt is cleaned by
# the garbage collector. This is useful for debugging or when KnowIt is used as a submodule of
# a larger codebase.

# KI = KnowIt()

# Providing a custom_exp_dir path means that the constructed instance of KnowIt will be
# associated with a static (potentially pre-existing) experiment output directory.
# This is useful for long-term experimentation.

KI = KnowIt(custom_exp_dir='/home/tian/postdoc_work/KnowIt_debugging/my_dummy_exp')

# ----------------------------------------------------------------------------------------------------------------------
# IMPORTING A CUSTOM ARCHITECTURE
# ----------------------------------------------------------------------------------------------------------------------

# KnowIt comes with a set of default architectures. However, if you want to import a custom one
# you can call this function. The custom arch script will be checked and saved in the
# experiment output directory. See the archs.arch_how_to.md file along with the three default
# architectures for more detail.
# KI.import_arch(custom_arch_path='/home/tian/postdoc_work/KnowIt_debugging/new_dummy_arch.py')

# ----------------------------------------------------------------------------------------------------------------------
# IMPORTING A DATASET
# ----------------------------------------------------------------------------------------------------------------------

# KnowIt has two default datasets for debugging. In order to handle external data you can import it
# into your experiment output directory as follows. See the data.raw_data_conversion.py script's heading
# for more details.

# import_args = {'path': '/home/tian/postdoc_work/LEAH_SYNTH_DATA/synth_1/synth_1.pickle',
#                'base_nan_filler': 'linear',
#                'nan_filled_components': ['x1', 'x2', 'x3', 'x4', 'y1']}
# KI.import_dataset({'data_import_args': import_args})

# ----------------------------------------------------------------------------------------------------------------------
# CHECK AVAILABLE DATASETS & ARCHITECTURES
# ----------------------------------------------------------------------------------------------------------------------

# check the state of global arguments for current instance of KnowIt
print(KI.global_args())
# check the available datasets for current instance of KnowIt
print(KI.available_datasets())
# check the available architectures for current instance of KnowIt
print(KI.available_archs())
# check the metadata of a particular dataset for current instance of KnowIt
print(KI.summarize_dataset('synth_1'))

exit(101)

# ----------------------------------------------------------------------------------------------------------------------
# TRAINING A MODEL
# ----------------------------------------------------------------------------------------------------------------------
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
                'max_epochs': 3,
                'learning_rate': 0.01,
                'task': 'regression'
                }
KI.train_model(model_name=model_name, args={'data': data_args, 'arch': arch_args, 'trainer': trainer_args})


# KI.train_model_further(model_name=model_name, max_epochs=6)

# ----------------------------------------------------------------------------------------------------------------------
# GENERATE MODEL PREDICTIONS
# ----------------------------------------------------------------------------------------------------------------------

# model_name = "my_new_model"
# KI.generate_predictions(model_name=model_name, args={'predictor': {'prediction_set': 'train'}})
# KI.generate_predictions(model_name=model_name, args={'predictor': {'prediction_set': 'valid'}})
# KI.generate_predictions(model_name=model_name, args={'predictor': {'prediction_set': 'eval'}})

# ----------------------------------------------------------------------------------------------------------------------
# INTERPRET MODEL PREDICTIONS
# ----------------------------------------------------------------------------------------------------------------------

# model_name = "my_new_model"
#
# interpret_args = {'interpretation_method': 'DeepLift',
#                   'interpretation_set': 'eval',
#                   'selection': 'success',
#                   'size': 100}
#
# KI.interpret_model(model_name=model_name, args={'interpreter': interpret_args})

exit(101)
