"""
---------------------
ClassificationDataset
---------------------
This module represents a ``PreparedDataset`` that has the ability to create a Pytorch
dataloader ready for classification tasks. It inherits from ``PreparedDataset``.

In addition to all the functionality of ``PreparedDataset``, it also determines the class set
in the data. It assumes that each unique state of the output components is a single class.
It also changes the ``PreparedDataset.out_shape`` to (1, c), where c is the number of classes.
It also changes ``PreparedDataset.scaling_tag`` to "in_only" if "full" is selected.

Additionally, the ``ClassificationDataset.get_dataloader`` function
extracts the corresponding dataset split, casts it as a ``CustomClassificationDataset``
object, and returns a Pytorch DataLoader from it.

---------------------------
CustomClassificationDataset
---------------------------

This is a custom dataset that inherits from the Pytorch Dataset class.
It receives the full input (x) and output (y) arrays as arguments, as well as the class set (c).
It casts the y-values into integer classes according to the class set.
When an item is sampled with __getitem__, the relevant sample is taken from x and y,
x is cast as a Tensor with float type, and the unique sample index is also returned.
"""
from __future__ import annotations
__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the ClassificationDataset class for KnowIt.'

# external imports
from numpy import isnan, unique, zeros, argwhere, array
from torch.utils.data import Dataset, DataLoader
from torch import is_tensor, from_numpy, Tensor
from torch import zeros as zeros_tensor

# internal imports
from data.prepared_dataset import PreparedDataset
from helpers.logger import get_logger
logger = get_logger()


class ClassificationDataset(PreparedDataset):
    """The ClassificationDataset class inherits from the PreparedDataset class
    and is used to perform classification tasks.

    This is the ClassificationDataset class that is used to create a classification
    specific KnowIt dataset. It contains all the attributes and functions in PreparedDataset,
    in addition to methods that can create a Pytorch dataloader ready for training
    a classification model along with the following.

    Parameters
    ----------
    **kwargs : dict[str, any]
        Keyword arguments that are passed to the PreparedDataset constructor.

    Attributes
    ----------
        class_set : dict[any, int]
            A dictionary that has class labels as keys and corresponding unique integers as values.
        class_counts : dict[any, int]
            A dictionary that has class labels as keys and corresponding
            class counts (number of prediction points) as values.

    Raises
    ------
    SystemExit
        If the `out_chunk` attribute does not have matching start and end values, the
        program will log an error message and exit with status code 101.
    """
    class_set = {}
    class_counts = {}

    def __init__(self, **kwargs) -> None:
        if kwargs['out_chunk'][0] != kwargs['out_chunk'][1]:
            logger.error('Currently, KnowIt can only perform classification at one specific time step at a time. '
                         'Please change the out_chunk %s argument to reflect this. Both values must match.',
                         str(kwargs['out_chunk']))
            exit(101)
        if kwargs['scaling_tag'] == 'full':
            logger.warning('scaling_tag cannot be full for classification tasks. Changing to scaling_tag=in_only.')
            kwargs['scaling_tag'] = 'in_only'
        super().__init__(**kwargs)
        self._get_classes()
        self.out_shape = [1, len(self.class_set)]

    def _get_classes(self) -> None:
        """Identify and count unique classes in the dataset.

        This method processes the dataset to determine the unique classes present in the data
        and counts the occurrences of each class. The unique classes and their counts are stored
        in the `class_set` and `class_counts` attributes, respectively.

        Notes
        -----
        The method uses the following steps.
            - The method retrieves the dataset using `get_the_data()`.
            - For each instance in `self.instances`, it iterates over the relevant slices.
            - It identifies the unique combinations of entries in the output components.
            - It counts the occurrences of each unique class and updates `self.class_counts`.
            - It updates `self.class_set` with the unique classes found.
        """
        the_data = self.get_the_data()
        found_class_set = set()
        for i in self.instances:
            for s in the_data[i]:
                vals = s['d'][:, self.y_map][~isnan(s['d'][:, self.y_map]).any(axis=1)]
                unique_entries = unique(vals, axis=0)
                unique_entries_list = []
                for u in unique_entries:
                    if len(u) > 1:
                        unique_entries_list.append(tuple(u))
                    else:
                        unique_entries_list.append(u.item())
                unique_entries = unique_entries_list
                for v in unique_entries:
                    new_count = len(argwhere(vals == v))
                    if v not in self.class_counts:
                        self.class_counts[v] = new_count
                    else:
                        self.class_counts[v] += new_count
                found_class_set.update(set(unique_entries))
        found_class_set = list(found_class_set)
        tick = 0
        for c in found_class_set:
            self.class_set[c] = tick
            tick += 1
        logger.info('Found %s unique classes.',
                    str(len(self.class_set)))
        logger.info(self.class_set)

    def get_dataloader(self, set_tag: str, *, analysis: bool = False, num_workers: int = 4) -> DataLoader:
        """Creates and returns a PyTorch DataLoader for a specified dataset split.

        This method generates a DataLoader for a given dataset split (e.g., train, valid, or eval).
        It uses the `CustomClassificationDataset` class to create the dataset and then initializes
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
        dataset = CustomClassificationDataset(self.class_set, **self.extract_dataset(set_tag))

        drop_last = False
        shuffle = False
        if set_tag == 'train' and not analysis:
            drop_last = True
            shuffle = self.shuffle_train

        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
        return dataloader


class CustomClassificationDataset(Dataset):
    """Custom dataset for PyTorch for classification tasks.

    This class creates a dataset format suitable for classification tasks, which can be used with
    PyTorch DataLoader. It handles the conversion of class labels to one-hot encoding and provides
    the necessary methods to get the length and individual samples of the dataset.

    Parameters
    ----------
    c : dict
        A dictionary mapping each unique class to a unique index.
    x : numpy.ndarray
        The input features of the dataset with shape (num_samples, num_features, ...).
    y : numpy.ndarray
        The class labels of the dataset with shape (num_samples, 1).

    Attributes
    ----------
    x : numpy.ndarray
        The input features of the dataset.
    y : numpy.ndarray
        The integer class labels of the dataset, transformed from the original class labels.
    c : dict
        The dictionary mapping each unique class to a unique index.
    """
    x = None
    y = None
    c = None

    def __init__(self, c: dict, x: array, y: array) -> None:
        self.x = x
        self.c = c
        # Initialize the labels array with -1 indicating an invalid state
        self.y = zeros(shape=y.shape[0]).astype(int) - 1
        # Convert class labels to integer indices
        for k, v in self.c.items():

            # assuming one output component
            # self.y[y.squeeze() == k] = v

            # got_class = (y == k).all(axis=1)
            # # self.y[got_class] = v
            # self.y[got_class.squeeze()] = v

            got_class = (y.squeeze(axis=1) == k).all(axis=1)
            self.y[got_class.squeeze()] = v

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

        output_y = self.y[idx]
        new_output_y = zeros_tensor(len(self.c))
        new_output_y[output_y] = 1
        output_y = new_output_y

        input_x = input_x.astype('float')
        sample = {'x': from_numpy(input_x).float(),
                  'y': output_y, 's_id': idx}
        return sample
