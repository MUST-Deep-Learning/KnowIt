Main author = randlerabe@gmail.com 

This subdirectory contains the scripts for managing model training and check-
pointing. We will refer to all modules in this subdirectory as the "trainer
framework" collectively.

Given a Pytorch model and Pytorch dataloaders (generated from KnowIt's data
framework), the trainer framework will train a model, log various metrics, and
generate a Pytorch Lightning checkpoint file.

The trainer framework design pattern is similar to a State pattern. In partic-
ular, the pattern consists of a context class that interfaces with KnowIt's
main architecture module, an abstract base class that defines abstract methods
and can act as an interface, and a set of trainer states that inherits from the
abstract base class.

The benefits of this design pattern is that it allows for easier debugging and
will allow the trainer framework to be further customizible by a user through 
incorporating custom trainer states.

**1. Modules**





