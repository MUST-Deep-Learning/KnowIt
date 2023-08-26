__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the prepared dataset class.'

from data.base_dataset import BaseDataset

class PreparedDataset(BaseDataset):

    def __init__(self, name: str,
                 split_method: str,
                 scaling_method: str,
                 sampling_method: str,
                 min_chunk: int,
                 augment_method: str,
                 dim_reduction_method: str):

        super().__init__(name, 'option')

        self.split_method = split_method
        self.scaling_method = scaling_method
        self.sampling_method = sampling_method
        self.min_chunk = min_chunk
        self.augment_method = augment_method
        self.dim_reduction_method = dim_reduction_method


# DS = PreparedDataset(name='dummy_zero', split_method='chrono',
#                scaling_method='z', sampling_method='random',
#                min_chunk=10, augment_method='None',
#                dim_reduction_method='None')
# ping = 0

