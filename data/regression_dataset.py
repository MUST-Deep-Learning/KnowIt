from torch.utils.data import Dataset, DataLoader
from torch import is_tensor, from_numpy

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
        output_y = output_y.squeeze()
        sample = {'x': from_numpy(input_x).float(),
                  'y': from_numpy(output_y).float()}
        return sample
