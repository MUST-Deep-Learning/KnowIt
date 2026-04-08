KnowIt uses deep learning architectures derived from [``torch.nn.Module``](https://pytorch.org/docs/stable/generated/torch.nn.Module.html).
This guide explains how to define, import, and manage new custom architectures. 
For details on importing datasets see the `Dataset How-to` guide.

## 1. Importing new architectures using Pytorch

To use an external architecture, a new python script must be written that meets 
a number of criteria. 
This script can then be imported using the ``KnowIt.import_arch(new_arch_path=<path to script>)`` method.
The imported architecture, which will have the same name as the imported script, will be stored under ``/custom_archs`` in the relevant custom experiment output directory.
It can then be used to train a model by passing ``kwargs={'arch': {'name': <architecture name>, ...}, ...}`` when 
calling the ``KI.train_model`` function.

The criteria are as follows:
1. The script must contain a global ``available_tasks (tuple)`` variable that contains strings defining what type of tasks are currently supported by the architecture. KnowIt currently supports ``(regression, classification, vl_regression)``.
2. The script must contain a global ``HP_ranges_dict (dict)`` variable of iterables that provide reasonable ranges for each hyperparameter related to the architecture. This variable currently only serves as information to potential users, but we might use this later for tuning purposes.
3. The script must contain a class called ``Model`` that inherits from [``torch.nn.Module``](https://pytorch.org/docs/stable/generated/torch.nn.Module.html).
    -   ``Model``'s constructor must take the following arguments:
        - **task_name** (str) = The task to be performed by the resulting model. 
        - **input_dim** (list) = The expected input dimensions of the model `[t_in, c_in]`. 
        - **output_dim** (list) = The expected output dimensions of the model `[t_out, c_out]`.
        - Where
          - if *task_name* = 'regression', then `t_in` and `t_out` is the number of input and output time steps (delays) and `c_in` and `c_out` is the number of input and output components, respectively.
          - if *task_name* = 'classification', then `t_in` is the number of input time steps (delays), `t_out=1` and `c_out` is the number of classes.
          - if *task_name* = 'vl_regression' then `t_in=t_out=1` and `c_in` and `c_out` is the number of input and output components, respectively.
    - All other arguments are optional, meant to correspond to specific hyperparameters, and default values must be defined.
    -   ``Model`` must also have a ``forward`` method, representing the forward function of the model, that receives, 
        - if *task_name* = 'regression', a Tensor of shape `[batch_size, t_in, c_in]` as argument, and produce a Tensor of shape `[batch_size, t_out, c_out]` as output.
        - if *task_name* = 'classification', a Tensor of shape `[batch_size, t_in, c_in]` as argument, and produce a Tensor of shape `[batch_size, num_classes]` as output.
        - if *task_name* = 'vl_regression', a Tensor of shape `[batch_size, t, c_in]` as argument, and produce a Tensor of shape `[batch_size, t, c_out]` as output, where `t` is variable length.

See default architectures for examples.

## 2. Facilitating statefulness

Some architectures are meant to be stateful. This means that they carry an internal state across forward passes.
If this is desired a number of additional methods need to be defined.

An ``update_states(ist_idx, device)`` method signals to the architecture that the internal states should be updated based 
on the provided arguments:
 - **ist_idx** is a Tensor containing the IST indices (of shape `[batch_size, 3]`) of the next batch that will pass through the forward function.
 - **device** is a string defining the device (e.g., 'cuda' or 'cpu') on which the next batch will be located.
If defined, KnowIt will call this method at the start of each batch.
Additionally, some interpreters can call it to help manage the statefulness throughout interpretations.
By keeping track of the ``ist_idx`` values between calls of ``update_states(ist_idx, device)`` the architecture can reset 
only the part of the internal state that corresponds to breaks in contiguousness.

A ``hard_set_states(ist_idx)`` should set the current ``ist_idx`` values to the given ones.
If defined, KnowIt will call this method at the start of train, validation, and testing loops, 
right after ``update_states(ist_idx, device)`` is called. 
This function is only relevant if variable length data is modelled and statefulness is desired.
It is required to move the IST indices to the end of the variable length batch.

A ``force_reset()`` method signals to the architecture that all internal states should be reset the next time 
``update_states(ist_idx, device)`` is called regardless of contiguousness.
If defined, KnowIt will call this method at the start of train, validation, and testing loops.
Additionally, some interpreters can call it to help manage the statefulness throughout interpretations.

A ``get_internal_states()`` method requests that the architecture returns the current internal states in a specific format.
This is used to facilitate interpreters like [DeepLift](https://captum.ai/api/deep_lift.html) and 
[IntegratedGradients](https://captum.ai/api/integrated_gradients.html). Any internal states should be 
returned as a list of tensors (or a list of a list of tensors) where the first dimension in the tensors 
must correspond to the "batch index".

Finally, the ``forward`` method should receive an optional second argument ``*internal_states`` which is 
has the same structure at that which is returned by ``get_internal_states()``. If this argument is 
provided, the current internal states of the architecture should be overwritten with it.

The modifications are only required if statefulness is required. In all cases KnowIt will check 
if the architecture has the required methods before calling them. See the default architecture ``KnowIt/default_archs/LSTMv2`` 
for an example of an architecture that can be stateful or stateless, and manages the internal states appropriately.


## 3. Useful functions

Use ``KI.available_archs()`` to see what architectures are available after importing.

## 4. Default architectures

While newly imported architectures are stored under ``/custom_archs`` in the relevant custom experiment output directory, 
default datasets are stored under ``KnowIt/default_archs``. There are currently six default architectures. 

 - MLP - A Multilayer perceptron architecture. Supports regression and classification.
 - LSTM - A Long short-term memory architecture. Supports regression and classification.
 - TCN - A Temporal convolutional architecture. Supports regression, classification, and vl_regression.
 - CNN - A 1D convolutional architecture. Supports regression, classification, and vl_regression.
 - LSTMv2 - A Long short-term memory architecture that can also be stateful. Supports regression, classification, vl_regression, and stateful training.
 - TFT - A Temporal Fusion Transformer-style architecture that wraps the LSTMv2 architecture. Supports regression, classification, vl_regression, and stateful training.

