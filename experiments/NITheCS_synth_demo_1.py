from knowit import KnowIt

# ----------------------------------------------------------------------------------------------------------------------
# IMPORTING A DATASET
# ----------------------------------------------------------------------------------------------------------------------

# raw_data_path = '/home/tian/postdoc_work/NITheCS_demo_2023/dummy_raw_data/dummy_zero/dummy_zero.pickle'
# from helpers.read_configs import load_from_path
# my_dataframe = load_from_path(raw_data_path)
# import_args = {'path': raw_data_path,
#                'base_nan_filler': 'linear',
#                'nan_filled_components': ['x1', 'x2', 'x3', 'x4', 'y1', 'y2']}
# KnowIt(action='import', args={'import': import_args})

# ----------------------------------------------------------------------------------------------------------------------
# TRAINING A MODEL
# ----------------------------------------------------------------------------------------------------------------------

# id_args = {'experiment_name': 'NITheCS_synth_demo_1',
#            'model_name': 'my_first_model'}
# data_args = {'data': 'dummy_zero',
#              'task': 'regression',
#              'in_components': ['x1', 'x2', 'x3', 'x4'],
#              'out_components': ['y1', 'y2'],
#              'in_chunk': [-5, 5],
#              'out_chunk': [0, 0],
#              'split_portions': [0.6, 0.2, 0.2],
#              'batch_size': 64,
#              'split_method': 'instance-random',
#              'scaling_tag': 'full'}
# arch_args = {'task': 'regression',
#              'arch': 'MLP',
#              'arch_hps': {'dropout': 0.0,
#                           'width': 512}}
# trainer_args = {'loss_fn': 'mse_loss',
#                 'optim': 'Adam',
#                 'max_epochs': 50,
#                 'learning_rate': 0.001,
#                 'performance_metrics': 'mean_absolute_error'}
# KnowIt(action='train',
#        args={'id': id_args,
#              'data': data_args,
#              'arch': arch_args,
#              'trainer': trainer_args},
#        safe_mode=False)

# ----------------------------------------------------------------------------------------------------------------------
# EVALUATE MODEL PREDICTIONS
# ----------------------------------------------------------------------------------------------------------------------

# id_args = {'experiment_name': 'NITheCS_synth_demo_1',
#            'model_name': 'my_first_model'}
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
