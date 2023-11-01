__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the RegressionDataset class for Knowit.'

"""
-------------------
RegressionDataset
-------------------
This module represents a PreparedDataset that has the ability to create a Pytorch 
dataloader ready for regression tasks. It inherits from PreparedDataset.

The only difference is the addition of the RegressionDataset.get_dataloader function.
This function extracts the corresponding dataset split, casts it as a CustomRegressionDataset 
object, and creates a Pytorch DataLoader from it.


-------------------------
CustomRegressionDataset
-------------------------

This is a custom dataset that inherits from the Pytorch Dataset class.
It receives the full input (x) and output (y) arrays as arguments.
When an item is sampled with __getitem__, the relevant sample is taken from x and y,
each value is cast as a Tensor with float type, and the unique sample index is also returned.

"""

# external imports
import torch
from torch.utils.data import Dataset, DataLoader
from torch import is_tensor, from_numpy

# internal imports
from data.prepared_dataset import PreparedDataset


class RegressionDataset(PreparedDataset):

    def __init__(self, **args):
        super().__init__(**args)

    def get_dataloader(self, set_tag):
        dataset = CustomRegressionDataset(**self.extract_dataset(set_tag))

        drop_last = False
        if set_tag == 'train':
            drop_last = True

        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=self.shuffle_train, drop_last=drop_last)
        return dataloader


class CustomRegressionDataset(Dataset):
    """Dataset format for torch."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
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
