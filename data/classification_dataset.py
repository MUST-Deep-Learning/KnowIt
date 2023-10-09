from numpy import isnan, unique, zeros_like
from torch.utils.data import Dataset, DataLoader
from torch import is_tensor, from_numpy

from data.prepared_dataset import PreparedDataset


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
                unique_entries = unique(s['d'][:, self.y_map][~isnan(s['d'][:, self.y_map])], axis=0)
                unique_entries = [u for u in unique_entries]
                found_class_set.update(set(unique_entries))
        found_class_set = list(found_class_set)
        self.class_set = {}
        tick = 0
        for c in found_class_set:
            self.class_set[c] = tick
            tick += 1

    def get_dataloader(self, set_tag):
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
        self.y = zeros_like(y).astype(int)
        self.y = self.y.squeeze()
        self.c = c

        for k, v in self.c.items():
            self.y[y.squeeze() == k] = v

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
