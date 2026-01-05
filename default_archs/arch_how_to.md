KnowIt uses deep learning architectures derived from [``torch.nn.Module``](https://pytorch.org/docs/stable/generated/torch.nn.Module.html).
This guide explains how to import and manage new architectures. For details on the imported data structure see 
`KnowIt.data.base_dataset`.

## 1. Importing new architectures using Pytorch

To use an external architecture a new python script must be written that meets 
a number of criteria. This script can then be imported using the ``KnowIt.import_arch(new_arch_path=<path to script>)`` function.

The criteria are as follows:
1. The script must contain an ``available_tasks (tuple)`` variable that contains strings defining what type of task is currently supported by the model. KnowIt currently supports ``(regression, classification)``.
2. An ``HP_ranges_dict (dict)`` variable that provides the ranges of each hyperparameter, for tuning purposes.
3. A class called ``Model`` that inherits from [``torch.nn.Module``](https://pytorch.org/docs/stable/generated/torch.nn.Module.html).
    -   This class must take the following arguments:
        - **task_name** (str) = The current task to be performed by the model. 
        - **input_dim** (tuple) = The expected input dimensions of the model (t, c). Where t is the number of time steps (delays) and c is the number of input components. 
        - **output_dim** (tuple) = The expected output dimensions of the model (t, c). Where t is the number of time steps (delays) and c is the number of output components.
    - If the model is to perform classification, `output_dim` will be of the shape `[1, num_of_classes]`.
    - All other arguments are optional and default values must be provided.
    -   The class must also have a forward function that receives a Tensor of shape `[batch_size, in_chunk[b] - in_chunk[a], num_in_components]`, as argument, and produce a Tensor of shape `[batch_size, out_chunk[b] - out_chunk[a], num_out_components]`.
    - In the case of classification, the forward function must produces an output Tensor of shape `[batch_size, num_classes]`.
    - Additionally, if statefulness is desired, the following two methods must also be defined:
      - ``force_reset()``, accessable from outer scope. This method should reset all hidden and internal states. It is called at the start of train, validation, and testing loops.
      - ``update_states()`` it receives a Tensor of current IST indices of shape `[batch_size, 3]` as argument. This method can be used to handle statefulness. It is called at the start of each batch.

See default architectures for examples.

The imported architecure will be stored under ``/custom_archs`` in the relevant custom experiment output directory.
It can then be used to train a model by passing ``kwargs={'arch': {'name': <architecture name>, ...}, ...}`` when 
calling the ``KI.train_model`` function.

## 2. Useful functions

Use ``KI.available_archs()`` to see what architectures are available after importing.

## 3. Default architectures

While newly imported architectures are stored under ``/custom_archs`` in the relevant custom experiment output directory, 
default datasets are stored under ``KnowIt/default_archs``. There are currently four default architectures. 

 - MLP - A Multilayer perceptron architecture.
 - LSTM - A Long short-term memory architecture.
 - TCN - A Temporal convolutional architecture.
 - CNN - A 1D convolutional architecture.

