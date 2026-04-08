Main author = randlerabe@gmail.com 

This subdirectory contains the scripts for model interpretability techniques. We will refer to all modules in this subdirectory as the "interpreter framework" collectively.

Given a Pytorch model and a KnowIt datamodule, the interpreter framework will extract interpretations from the model relative to a user's data.

Currently, KnowIt's interpreter framework supports feature attribution algorithms from Captum's library. Given a datapoint or a set of datapoints (referred to as "prediction points"), feature attribution techniques assign an importance score to the input features of the model relative to each of the model's output features or classes.

Three feature attributions techniques have been implemented in KnowIt through the Captum library, namely, integrated gradients, deeplift, and deepliftshap. Feature attributions as well as plots is saved in a user's project directory.

**1. Modules**

The interpreter framework consists of the following modules (indentation indicates inheritance):
- KIInterpreter
    - FeatureAttribution
        - IntegratedGrad
        - DeepL
        - DLS

**2. KIInterpreter**

The _KIInterpreter_ module is the base class for the interpreter framework. Any interpretability class must inherit from _KIInterpreter_. The module initializes a model from a Pytorch Lightning checkpoint file. It stores the model as well as the datamodule (from KnowIt's data framework) and the hardware device to be used (gpu or cpu).

**3. FeatureAttribution**

The _FeatureAttribution_ module is an intermediate class that defines methods needed for the family of feature attribution algorithms. In particular, it implements a method that extracts data points from KnowIt's datamodule. The descendent classes uses the method to build baselines and extract the prediction points for which a user wants to generate attributions.

It inherits from _KIInterpreter_.

**4. IntegratedGrad**

The module _IntegratedGrad_ implements the integrated gradients technique. Integrated gradients belong to the family of feature attributions algorithms. See the module for more information. The algorithm can be applied to both regression and classification tasks.

It inherits from _FeatureAttributions_.

**5. DeepL**

The module _DeepL_ implements the deeplift technique. Deeplift belong to the family of feature attributions algorithms. See the module for more information. The algorithm can be applied to both regression and classification tasks.

It inherits from _FeatureAttributions_.

**6. DLS**

The module _DLS_ implements the deepliftshap technique. Deepliftshap belong to the family of feature attributions algorithms. See the module for more information. The algorithm can be applied to both regression and classification tasks.

It inherits from _FeatureAttributions_.
