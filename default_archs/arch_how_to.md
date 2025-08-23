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
1. The script must contain a global ``available_tasks (tuple)`` variable that contains strings defining what type of tasks are currently supported by the architecture. KnowIt currently supports ``(regression, classification)``.
2. The script must contain a global ``HP_ranges_dict (dict)`` variable of iterables that provide reasonable ranges for each hyperparameter related to the architecture. This variable currently only serves as information to potential users, but we might use this later for tuning purposes.
3. The script must contain a class called ``Model`` that inherits from [``torch.nn.Module``](https://pytorch.org/docs/stable/generated/torch.nn.Module.html).
    -   ``Model``'s constructor must take the following arguments:
        - **task_name** (str) = The task to be performed by the resulting model. 
        - **input_dim** (list) = The expected input dimensions of the model `[t, c]`. Where t is the number of time steps (delays) and c is the number of input components. 
        - **output_dim** (list) = The expected output dimensions of the model `[t, c]` (or `[1, num_of_classes]` for classification models). Where t is the number of time steps (delays) and c is the number of output components.
    - All other arguments are optional, meant to correspond to specific hyperparameters, and default values must be defined.
    -   ``Model`` must also have a ``forward`` method, representing the forward function of the model, that receives a Tensor of shape `[batch_size, in_chunk[b] - in_chunk[a], num_in_components]` as argument, and produce a Tensor of shape `[batch_size, out_chunk[b] - out_chunk[a], num_out_components]` (or `[batch_size, num_classes]` for classification models) as output.
    - Additionally, if statefulness is desired, the following two methods must also be defined:
      - ``force_reset()``, accessable from outer scope. This method should reset all hidden and internal states. It is called at the start of train, validation, and testing loops.
      - ``update_states()`` it receives a Tensor of current IST indices of shape `[batch_size, 3]` as argument. This method can be used to handle statefulness. It is called at the start of each batch.

See default architectures for examples.

## 2. Facilitating statefulness

Some architectures are meant to be stateful. This means that they carry an internal state across samples and batches.
If this is desired a number of additional methods need to be defined.

An ``update_states(ist_idx, device)`` method signals to the architecture that the internal states should be updated based 
on the provided arguments:
 - **ist_idx** is a Tensor containing the IST indices (of shape `[batch_size, 3]`) of the next batch that will pass through the forward function.
 - **device** is a string defining the device (e.g., 'cuda' or 'cpu') on which the next batch will be located.
If defined, KnowIt will call this method at the start of each batch.
Additionally, some interpreters can call it to help manage the statefulness throughout interpretations.
By keeping track of the ``ist_idx`` values between calls of ``update_states(ist_idx, device)`` the architecture can reset 
only the part of the internal state that corresponds to breaks in contiguousness.

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
if the argument has the required methods before calling them. See the default architecture ``KnowIt/default_archs/LSTMv2`` 
for an example of an architecture that can be stateful or stateless, and manages the internal states appropriately.



## 3. Useful functions

Use ``KI.available_archs()`` to see what architectures are available after importing.

## 4. Default architectures

While newly imported architectures are stored under ``/custom_archs`` in the relevant custom experiment output directory, 
default datasets are stored under ``KnowIt/default_archs``. There are currently four default architectures. 

 - MLP - A Multilayer perceptron architecture.
 - LSTM - A Long short-term memory architecture.
 - TCN - A Temporal convolutional architecture.
 - CNN - A 1D convolutional architecture.
 - LSTMv2 - A Long short-term memory architecture that can also be stateful.

