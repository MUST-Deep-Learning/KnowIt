Main author = randlerabe@gmail.com 

This subdirectory contains the scripts for managing model training and checkpointing. We will refer to all modules in this subdirectory as the "trainer framework" collectively.

Given a Pytorch model and Pytorch dataloaders (generated from KnowIt's data framework), the trainer framework will train a model, log various metrics, and generate a Pytorch Lightning checkpoint file.

The trainer framework design pattern is similar to a State pattern. In particular, the pattern consists of a context class that interacts with KnowIt's main architecture module, an abstract base class that defines abstract methods and can act as an interface between the context class and states, and a set of trainer states that inherits from the abstract base class.

The benefits of this design pattern are that it allows for easier debugging and will allow the trainer framework to be further customizable by a user through incorporating custom trainer states.

**1. Modules**

The trainer framework consists of the following modules (indentation indicates inheritance):
- BaseTrainer
    - TrainNew
    - ContinueTraining
    - EvaluateOnly
    - CustomTrainer
- KITrainer
- PLModel

**2. BaseTrainer**

The _BaseTrainer_ module is an abstract class that stores a user's input parameters and defines a set of abstract methods needed for the trainer states. It is inherited by one of the trainer states and functions similar to an interface between _KITrainer_ and a state. More information can be found in the modules documentation in _base_trainer.py_. 

**3. TrainNew**

A state that initializes and prepares the KnowIt trainer to train a new model, log various metrics, and save a model checkpoint file. The metrics and checkpoint file will be automatically saved to a user's project directory.

_TrainNew_ inherits attributes and methods from _BaseTrainer_.

**4. ContinueTraining**

A state that initializes and prepares the KnowIt trainer to continue training an existing model from a checkpoint. Similar to _TrainNew_, this state module with log various metrics and save a new model checkpoint file. The metrics and checkpoint file will be automatically saved to a user's project directory.

_ContinueTraining_ inherits attributes and methods from _BaseTrainer_.

**5. EvaluateOnly**

A state that initializes and prepares the KnowIt trainer to evaluate an existing model from checkpoint on a set of dataloaders. The results will be printed in the terminal.

_EvaluateOnly_ inherits attributes and methods from _BaseTrainer_.

**6. CustomTrainer**

A custom template that can be edited by a user for more niche applications that are covered by the above states.

The custom class must inherit attributes and methods from _BaseTrainer_.

**7. KITrainer**

The _KITrainer_ module interacts with KnowIt's architecture script. Based on a user's task, the module will prepare the trainer in one of the states. After training is completed, the model will be evaluated on a user's dataloaders and the results printed in the terminal.


**8. PLModel**

The _PLModel_ module is a wrapper class that takes a user's Pytorch model class and any required parameters and builds a Pytorch Lightning model. The Pytorch Lightning model defines all the necessary methods required by Pytorch Lightning's Trainer module.






