Main author = tiantheunissen@gmail.com

This directory contains the scripts for managing datastructures in KnowIt.
We call these scripts the "data framework" or "data module" collectively.

The data framework creates custom datasets that 
are stored in the 'custom_datasets' subdirectory of the experiment output directory. 
It then loads this data and prepares it for other modules in KnowIt.

**1. Modules**

The data framework of KnowIt consists of two classes with the following inheritance structure:
- BaseDataset
  - PreparedDataset(BaseDataset)

**2. BaseDataset**

The _BaseDataset_ represents the bare minimum info KnowIt requires about the data.
It serves to load, clean, compile, and store raw data (with critical metadata).
It does not carry any concept of models or training, only information related
to the data is stored. See the top of the _base_dataset.py_ 
script for more details.

**3. PreparedDataset(BaseDataset)**

The _PreparedDataset_ represents a dataset that is preprocessed for model training.
It contains all the variables in _BaseDataset_, 
in addition to metadata regarding data splitting, scaling, sampling, 
and shuffling. See the top of the _prepared_dataset.py_ script for more details.
It also contains methods that can create a Pytorch dataloader ready for training or interpreting.
Depending on the task, the dataloader uses one of the following datasets:
- CustomDataset(torch.utils.data.Dataset)
  - CustomClassificationDataset(CustomDataset)

**Additional sub-modules**

- RawDataConverter: Used by BaseDataset to create new datasets from a given pandas Dataframe.
- DataSplitter: Used by PreparedDataset to split the raw data into train/valid/eval sets. 
  This is defined with a set of three "selection matrices".
- DataScaler: Used by PreparedDataset to scale the raw data for model training.

See the corresponding scripts for details.

