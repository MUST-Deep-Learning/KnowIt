from knowit import KnowIt

KI = KnowIt(custom_exp_path='/home/tian/postdoc_work/KnowIt_debugging/my_dummy_exp_classification')

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
                'max_epochs': 3,
                'learning_rate': 0.001,
                'learning_rate_scheduler': {'ExponentialLR': {'gamma': 0.9}},
                'task': 'classification'}
KI.train_model(model_name=model_name, args={'data': data_args, 'arch': arch_args, 'trainer': trainer_args})

# ----------------------------------------------------------------------------------------------------------------------
# GENERATE MODEL PREDICTIONS
# ----------------------------------------------------------------------------------------------------------------------

model_name = "my_new_penguin_model"
KI.generate_predictions(model_name=model_name, args={'predictor': {'prediction_set': 'train'}})
KI.generate_predictions(model_name=model_name, args={'predictor': {'prediction_set': 'valid'}})
KI.generate_predictions(model_name=model_name, args={'predictor': {'prediction_set': 'eval'}})

# ----------------------------------------------------------------------------------------------------------------------
# INTERPRET MODEL PREDICTIONS
# ----------------------------------------------------------------------------------------------------------------------

model_name = "my_new_penguin_model"

interpret_args = {'interpretation_method': 'DeepLift',
                  'interpretation_set': 'eval',
                  'selection': 'success',
                  'size': 100}

KI.interpret_model(model_name=model_name, args={'interpreter': interpret_args})




exit(101)