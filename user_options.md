The user interacts with KnowIt by providing relevant keyword arguments (kwargs) to specific functions.
Some functions require one or more kwarg dictionaries, which have required and optional arguments.
In the absence of optional arguments, defaults will be provided. These defaults are set in 
`KnowIt.setup.setup_action_args`.

This page provides a summary of the main four functions used to interact with KnowIt.
See the relevant API docs for more details. See the tutorials for an example of these functions in action.

---

## 1. Importing a new dataset

 - **relevant functions**: ``KnowIt.import_dataset``
 - **kwarg keys**: 'data_import'
 - **relevant modules**: ``KnowIt.data.raw_data_conversion.RawDataConverter``
 - **relevant docs**: Dataset How-to

### Example
The following code constructs a KnowIt object, linked to an experiment output directory, 
and calls the ``KnowIt.import_dataset`` function with the relevant kwargs.

```python
from knowit import KnowIt
KI = KnowIt('/my_experiment_dir')
import_args = {'path': '/my_new_raw_data.pickle',
               'base_nan_filler': 'linear',
               'nan_filled_components': ['x1', 'x2', 'x3']}
KI.import_dataset(kwargs={'data_import': import_args})
```

---

## 2. Importing a new architecture

 - **relevant functions**: ``KnowIt.import_arch``
 - **kwarg keys**: None
 - **relevant modules**: ``KnowIt.setup.import_custom_arch.import_custom_arch``
 - **relevant docs**: Dataset How-to

### Example
The following code constructs a KnowIt object, linked to an experiment output directory, 
and calls the ``KnowIt.import_arch`` function with the relevant argument.

```python
from knowit import KnowIt
KI = KnowIt('/my_experiment_dir')
KI.import_arch('/my_new_arch.py')
```

---

## 3. Training a model

 - **relevant functions**: ``KnowIt.train_model``
 - **kwarg keys**: 'data', 'arch', 'trainer'
 - **relevant modules**: ``KnowIt.data.prepared_dataset.PreparedDataset``, 
``KnowIt.trainer.base_trainer.BaseTrainer``, ``KnowIt.trainer.trainer_states.TrainNew``
 - **relevant docs**: None

### Example
The following code constructs a KnowIt object, linked to an experiment output directory, 
and calls the ``KnowIt.train_model`` function with the relevant kwargs.

```python
from knowit import KnowIt
KI = KnowIt('/my_experiment_dir')
model_name = "my_new_model_name"
data_args = {'name': 'synth_2',
             'task': 'regression',
             'in_components': ['x1', 'x2', 'x3', 'x4'],
             'out_components': ['x5'],
             'in_chunk': [-35, 0],
             'out_chunk': [0, 0],
             'split_portions': [0.6, 0.2, 0.2],
             'batch_size': 64,
             'split_method': 'instance-random',
             'scaling_tag': 'full'}
arch_args = {'task': 'regression',
             'name': 'MLP'}
trainer_args = {'loss_fn': 'mse_loss',
                'optim': 'Adam',
                'max_epochs': 10,
                'learning_rate': 0.01,
                'task': 'regression'
                }
KI.train_model(model_name=model_name, kwargs={'data': data_args, 'arch': arch_args, 'trainer': trainer_args})
```

---

## 3. Generate model predictions (to interpret)

 - **relevant functions**: ``KnowIt.generate_predictions``
 - **kwarg keys**: 'predictor'
 - **relevant modules**: None
 - **relevant docs**: None

### Example
The following code constructs a KnowIt object, linked to an experiment output directory, 
and calls the ``KnowIt.generate_predictions`` function with the relevant kwargs.

```python
from knowit import KnowIt
KI = KnowIt('/my_experiment_dir')
model_name = "my_mlp"
KI.generate_predictions(model_name=model_name, kwargs={'predictor': {'prediction_set': 'eval'}})
```

---

## 4. Interpret predictions

 - **relevant functions**: ``KnowIt.interpret_model``
 - **kwarg keys**: 'interpreter'
 - **relevant modules**: ``KnowIt._get_interpret_setup``, ``KnowIt.setup.select_interpretation_points.get_interpretation_inx``
 - **relevant docs**: None

### Example
The following code constructs a KnowIt object, linked to an experiment output directory, 
and calls the ``KnowIt.interpret_model`` function with the relevant kwargs.

```python
from knowit import KnowIt
KI = KnowIt('/my_experiment_dir')
model_name = "my_mlp"
interpret_args = {'interpretation_method': 'DeepLiftShap',
                  'interpretation_set': 'eval',
                  'selection': 'success',
                  'size': 100}
KI.interpret_model(model_name=model_name, kwargs={'interpreter': interpret_args})
```