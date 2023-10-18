__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the ClassificationDataset class for Knowit.'

"""
-------------------
ClassificationDataset
-------------------
This module represents a PreparedDataset that has the ability to create a Pytorch 
dataloader ready for classification tasks. It inherits from PreparedDataset.

In addition to all the functionality of PreparedDataset, it also determines the class set 
in the data. It assumes that each unique state of the output components is a single class.

Additionally, the ClassificationDataset.get_dataloader function 
extracts the corresponding dataset split, casts it as a CustomClassificationDataset 
object, and creates a Pytorch DataLoader from it.


------------------------------
CustomClassificationDataset
------------------------------

This is a custom dataset that inherits from the Pytorch Dataset class.
It receives the full input (x) and output (y) arrays as arguments, as well as the class set (c).
It casts the y-values into integer classes according to the class set.
When an item is sampled with __getitem__, the relevant sample is taken from x and y,
x is cast as a Tensor with float type, and the unique sample index is also returned.

"""

# external imports
from numpy import isnan, unique, zeros
from torch.utils.data import Dataset, DataLoader
from torch import is_tensor, from_numpy

# internal imports
from data.prepared_dataset import PreparedDataset
from helpers.logger import get_logger

logger = get_logger()


class ClassificationDataset(PreparedDataset):

    def __init__(self, **args):
        super().__init__(**args)
        self.class_set = None
        self.__get_classes()

    def __get_classes(self):

        the_data = self.get_the_data()
        found_class_set = set()
        for i in self.instances:
            for s in the_data[i]:

                # assuming one output component
                # unique_entries = unique(s['d'][:, self.y_map][~isnan(s['d'][:, self.y_map])], axis=0)
                # unique_entries = [u for u in unique_entries]

                vals = s['d'][:, self.y_map][~isnan(s['d'][:, self.y_map]).any(axis=1)]
                unique_entries = unique(vals, axis=0)
                unique_entries = [tuple(u) for u in unique_entries]

                found_class_set.update(set(unique_entries))
        found_class_set = list(found_class_set)
        self.class_set = {}
        tick = 0
        for c in found_class_set:
            self.class_set[c] = tick
            tick += 1

        logger.info('Found %s unique classes.',
                    str(len(self.class_set)))
        logger.info(self.class_set)

    def get_dataloader(self, set_tag: str):
        dataset = CustomClassificationDataset(self.class_set, **self.extract_dataset(set_tag))

        drop_last = False
        if set_tag == 'train':
            drop_last = True

        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=self.shuffle_train, drop_last=drop_last)
        return dataloader


class CustomClassificationDataset(Dataset):
    """Dataset format for torch."""

    def __init__(self, c, x, y):
        self.x = x

        # assuming one output component
        # self.y = zeros_like(y).astype(int)
        # self.y = self.y.squeeze()

        self.y = zeros(shape=y.shape[0]).astype(int) - 1

        self.c = c

        for k, v in self.c.items():

            # assuming one output component
            # self.y[y.squeeze() == k] = v

            got_class = (y == k).all(axis=1)
            self.y[got_class] = v

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()
        input_x = self.x[idx, :, :]
        output_y = self.y[idx]
        input_x = input_x.astype('float')
        sample = {'x': from_numpy(input_x).float(),
                  'y': output_y, 's_id': idx}
        return sample
