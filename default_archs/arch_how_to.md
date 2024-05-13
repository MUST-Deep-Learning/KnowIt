This directory contains the default deep learning architectures currently available in Knowit.
Each script represents a different architecture, sharing the same name as the script.

To use an external architecture a few details must be provided in the script:

    -   An available_tasks(tuple) variable that contains strings defining what type of task is currently supported by the model.
        Knowit currently supports (regression, classification).

    -   An HP_defaults_dict(dict) variable that defines the default value of hyperparameters.

    -   An HP_ranges_dict(dict) variable that provides the ranges of each hyperparameter, for tuning purposes.

    -   A class called 'Model' that inherits from torch.nn.Module.

            -   This class must take the following arguments at init:

                    -   task_name(str) = The current task to be performed by the model.
                    -   input_dim(tuple) = The expected input dimensions of the model (t, c). 
                            Where t is the number of time steps (delays) and c is the number of input components. 
                    -   output_dim(tuple) = The expected output dimensions of the model (t, c). 
                            Where t is the number of time steps (delays) and c is the number of output components.
                All other arguments are optional and can be obtained from HP_defaults_dict.

            -   The class must also have a forward function.

            - Otherwise the only current expectation is that the module is based on pytorch modules.

A .py file meeting these conditions can be imported using the KnowIt.import_arch method. 
This will save a custom architecture in the relevant experiment output directory.

