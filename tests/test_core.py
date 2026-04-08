# Import the KnowIt module
import shutil
from knowit import KnowIt

# Instantiate a KnowIt object linked to an output directory
KI = KnowIt(custom_exp_dir='knowit_temp_test', safe_mode=False)

# Define your dataset
data_args = {'name': 'synth_2',
             'task': 'regression',
             'in_components': ['x1', 'x2', 'x3', 'x4'],
             'out_components': ['x5'],
             'in_chunk': [-32, 0],
             'out_chunk': [0, 0],
             'split_portions': [0.6, 0.2, 0.2],
             'batch_size': 64}

# Define your architecture
arch_args = {'task': 'regression',
             'name': 'MLP'}

# Define your trainer
trainer_args = {'loss_fn': 'mse_loss',
                'optim': 'Adam',
                'max_epochs': 1,
                'learning_rate': 0.01,
                'task': 'regression'}

# Train your model
KI.train_model(model_name='my_new_model_name',
               kwargs={'data': data_args,
                       'arch': arch_args,
                       'trainer': trainer_args}, safe_mode=False)

# Generate model predictions
KI.generate_predictions(model_name='my_new_model_name',
                        kwargs={'predictor': {'prediction_set': 'valid'}}, safe_mode=False)

# Interpret model predictions
KI.interpret_model(model_name='my_new_model_name',
                   kwargs={'interpreter':
                               {'interpretation_method': 'DeepLift',
                                'interpretation_set': 'valid',
                                'selection': 'random',
                                'size': 100}}, safe_mode=False)

shutil.rmtree(KI.exp_output_dir)