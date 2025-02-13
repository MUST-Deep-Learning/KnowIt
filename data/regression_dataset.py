"""
-----------------
RegressionDataset
-----------------
This module represents a ``PreparedDataset`` that has the ability to create a Pytorch
dataloader ready for regression tasks. It inherits from ``PreparedDataset``.

Additionally, the ``RegressionDataset.get_dataloader`` function
extracts the corresponding dataset split, casts it as a ``CustomRegressionDataset``
object, and returns a Pytorch DataLoader from it.

-----------------------
CustomRegressionDataset
-----------------------

This is a custom dataset that inherits from the Pytorch Dataset class.
It receives the full input (x) and output (y) arrays as arguments.
When an item is sampled with __getitem__, the relevant sample is taken from x and y,
each value is cast as a Tensor with float type, and the unique sample index is also returned.
"""
from __future__ import annotations
__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the RegressionDataset class for Knowit.'

# external imports
from torch.utils.data import Dataset, DataLoader
from torch import is_tensor, from_numpy, Tensor
from numpy import array

# internal imports
from data.prepared_dataset import PreparedDataset


class RegressionDataset(PreparedDataset):
    """The RegressionDataset class inherits from the PreparedDataset class and is used to perform regression tasks.

    This is the RegressionDataset class that is used to create a regression
    specific KnowIt dataset. It contains all the attributes and functions in PreparedDataset,
    in addition to methods that can create a Pytorch dataloader ready for training
    a regression model along with the following.

    Parameters
    ----------
    **kwargs : dict[str, any]
        Keyword arguments that are passed to the PreparedDataset constructor.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_dataloader(self, set_tag, analysis: bool = False, num_workers: int = 4) -> DataLoader:
        """Creates and returns a PyTorch DataLoader for a specified dataset split.

        This method generates a DataLoader for a given dataset split (e.g., train, valid, or eval).
        It uses the `CustomRegressionDataset` class to create the dataset and then initializes
        a DataLoader with the specified parameters.

        Parameters
        ----------
        set_tag : str
            A string indicating the dataset split to load ('train', 'valid', 'eval').
        analysis : bool, default = False
            A flag indicating whether the dataloader is being used for analysis purposes. If set to True,
            the `drop_last` and `shuffle` parameters of the DataLoader will be set to False.
        num_workers : int, default = 4
            Sets the number of workers to use for loading the dataset.

        Returns
        -------
        DataLoader
            A PyTorch DataLoader for the specified dataset split.

        Notes
        -----
        If `set_tag` is set to `valid` or `eval` then the `drop_last` and `shuffle` parameters of the DataLoader will
        be set to False.
        """
        dataset = CustomRegressionDataset(**self.extract_dataset(set_tag))

        drop_last = False
        shuffle = False
        if set_tag == 'train' and not analysis:
            drop_last = True
            shuffle = self.shuffle_train

        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
        return dataloader


class CustomRegressionDataset(Dataset):
    """Custom dataset for PyTorch for regression tasks.

    This class creates a dataset format suitable for regression tasks, which can be used with
    PyTorch DataLoader. It provides the necessary methods to get the length and individual samples of the dataset.

    Parameters
    ----------
    x : numpy.ndarray
        The input features of the dataset with shape (num_samples, num_features, ...).
    y : numpy.ndarray
        The class labels of the dataset with shape (num_samples, 1).

    Attributes
    ----------
    x : array
        The input features of the dataset.
    y : array
        The integer class labels of the dataset, transformed from the original class labels.
    """
    x = None
    y = None

    def __init__(self, x: array, y: array) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
        --------
        int
            The number of samples in the dataset.
        """
        return self.y.shape[0]

    def __getitem__(self, idx: int | Tensor) -> dict:
        """Return a single sample from the dataset at the given index.

        Parameters
        ----------
        idx : int or Tensor
            The index of the sample to retrieve.

        Returns
        -------
        dict[str, any]
            A dictionary containing
                -   'x' (Tensor): the input features,
                -   'y' (Tensor): the one-hot encoded labels,
                -   's_id' (int): the sample ID 's_id'.
        """
        if is_tensor(idx):
            idx = idx.tolist()
        input_x = self.x[idx, :, :]
        output_y = self.y[idx, :, :]
        input_x = input_x.astype('float')
        output_y = output_y.astype('float')
        sample = {'x': from_numpy(input_x).float(),
                  'y': from_numpy(output_y).float(),
                  's_id': idx}
        return sample
