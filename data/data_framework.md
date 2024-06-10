Main author = tiantheunissen@gmail.com

This directory contains the scripts for managing datastructures in KnowIt.
We call these scripts the "data framework" collectively.

Generally, the data framework creates custom datasets that 
are stored in the 'custom_datasets' subdirectory of the experiment output directory. 
It then loads this data and prepares it for other modules in KnowIt.

**1. Modules**

The data framework of KnowIt consists of four classes with the following inheritance structure:
- BaseDataset
  - PreparedDataset
    - RegressionDataset
    - ClassificationDataset

**2. BaseDataset**

The _BaseDataset_ represents the bare minimum info KnowIt requires.
It serves to load, clean, compile, and store raw data (with critical metadata).
It does not carry any concept of models or training, only information related
to the data is stored. See the top of the _base_dataset.py_ 
script for more details.

**3. PreparedDataset(BaseDataset)**

The _PreparedDataset_ represents a dataset that is preprocessed for model training.
It is meant to be an abstract class. It contains all the variables in _BaseDataset_, 
in addition to metadata regarding data splitting, scaling, sampling, 
and shuffling. See the top of the _prepared_dataset.py_ script for more details.

**4. RegressionDataset(PreparedDataset)**

The _RegressionDataset_ represents a dataset that is preprocessed and compiled
for regression model training. It contains all the variables in 
_PreparedDataset_, in addition to methods that can 
create a Pytorch dataloader ready for training a regression model. See the top 
of the _regression_dataset.py_ script for more details.

**5. ClassificationDataset(PreparedDataset)**

The _ClassificationDataset_ represents a dataset that is preprocessed and compiled
for classification model training. It contains all the variables in 
_PreparedDataset_, in addition to methods that can 
create a Pytorch dataloader ready for training a classification model. See the top 
of the _classification_dataset.py_ script for more details.

**Additional sub-modules**

- RawDataConverter: Used by BaseDataset to create new datasets from a given dataframe.
- DataSplitter: Used by PreparedDataset to split the raw data into train/valid/eval sets.
- DataScaler: Used by PreparedDataset to scale the raw data for model training.

See the corresponding scripts for details.

