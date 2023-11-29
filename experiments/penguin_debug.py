from knowit import KnowIt

# ----------------------------------------------------------------------------------------------------------------------
# TRAINING A MODEL
# ----------------------------------------------------------------------------------------------------------------------

id_args = {'experiment_name': 'Penguin_debug_exp',
           'model_name': 'my__model'}
# data_args = {'data': 'penguin_42_debug',
#              'task': 'classification',
#              'in_components': ['accX', 'accY', 'accZ'],
#              'out_components': ['PCE'],
#              'in_chunk': [-25, 25],
#              'out_chunk': [0, 0],
#              'split_portions': [0.6, 0.2, 0.2],
#              'batch_size': 32,
#              'split_method': 'chronological',
#              'scaling_tag': 'in_only',
#              'min_slice': 100}
# arch_args = {'task': 'classification',
#              'arch': 'CNN'}
# trainer_args = {'loss_fn': 'weighted_cross_entropy',
#                 'optim': 'Adam',
#                 'max_epochs': 3,
#                 'learning_rate': 0.001,
#                 'learning_rate_scheduler': {'ExponentialLR': {'gamma': 0.9}},
#                 'task': 'classification'}
# KnowIt(action='train',
#        args={'id': id_args,
#              'data': data_args,
#              'arch': arch_args,
#              'trainer': trainer_args}, safe_mode=False)

# ----------------------------------------------------------------------------------------------------------------------
# EVALUATE MODEL PREDICTIONS
# ----------------------------------------------------------------------------------------------------------------------
#
# predict_args = {'prediction_set': 'train'}
# KnowIt(action='predict',
#        args={'id': id_args,
#              'predict': predict_args})
# predict_args = {'prediction_set': 'valid'}
# KnowIt(action='predict',
#        args={'id': id_args,
#              'predict': predict_args})
# predict_args = {'prediction_set': 'eval'}
# KnowIt(action='predict',
#        args={'id': id_args,
#              'predict': predict_args})

from helpers.viz import set_predictions
set_predictions(id_args, 'eval')
set_predictions(id_args, 'valid')
set_predictions(id_args, 'train')


# ----------------------------------------------------------------------------------------------------------------------
# INTERPRETING A MODEL
# ----------------------------------------------------------------------------------------------------------------------

# id_args = {'experiment_name': 'NITheCS_synth_demo_1',
#            'model_name': 'my_first_model'}
# interpret_args = {'interpretation_method': 'DeepLiftShap',
#                   'interpretation_set': 'eval',
#                   'selection': 'success',
#                   'size': 100}
#
# KnowIt(action='interpret',
#        args={'id': id_args,
#              'interpret_args': interpret_args})
