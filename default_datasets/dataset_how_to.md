KnowIt uses a specific datastructure to process time series data. 
This guide explains how to import and manage new raw data.

## 1. Importing new datasets using Pandas

In order to train models on your data you will need to convert it into a specific format for 
KnowIt to understand. Your data will need to be compiled into a pickled 
[``pandas.Dataframe``](https://pandas.pydata.org/docs/reference/frame.html) that meets 
a number of criteria. It can then be imported using the ``KnowIt.import_dataset(kwarg={'import_data_args': {'path': <path to pickle>, ...}})`` function.

The criteria are as follows:
1. Must be time indexed. (with a [``pandas.DatetimeIndex``](https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex), not strings)
2. Must contain the following metadata in the [``Dataframe.attrs``](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.attrs.html#pandas.DataFrame.attrs) dictionary, or alternatively passed with the 'meta' argument.
     - **name** (str): The name of the dataset to be constructed.
     - **components** (list): The components to be stored in the datasets.
     - **time_delta** ([``pandas.Timedelta``](https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html), [``datetime.timedelta``](https://docs.python.org/3/library/datetime.html)): The time difference between any two consecutive time points.
     - **instances** (list, Optional): A list of the instances names to be stored in the datasets.
3. Must contain no all-NaN columns.
4. Must contain column headers corresponding to the components defined in the metadata.


If instances are desired, they must be defined in the metadata and a corresponding 
column header 'instance' must be present in the dataframe. 
This column contains no NaNs, and indicates what instance each time step (row) corresponds to.
If no instances are define all time steps will be assumed to belong to one single instance.

The resulting datastructure will be stored under ``/custom_datasets`` in the relevant custom experiment output directory.
It can then be used to train a model by passing ``kwargs={'data': {'name': <data name>, ...}, ...}`` when 
calling the ``KI.train_model`` function.

## 2. Useful functions

Use ``KI.available_datasets()`` to see what datasets are available after importing.

Use ``KI.summarize_dataset('synth_2')`` to see a summary of your new dataset as it is imported.

## 3. Default datasets

While newly imported datasets are stored under ``/custom_datasets`` in the relevant custom experiment output directory, 
default datasets are stored under ``KnowIt/default_datasets``.