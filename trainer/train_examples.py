from data.classification_dataset import ClassificationDataset as clf
from data.base_dataset import BaseDataset

import pickle

def main():
    data_option = "penguin_pce_full"
    #bsd = BaseDataset.from_path(path="/home/randle/Dev/original_penguin_pce_full.pkl", safe_mode=False, base_nan_filler='split', nan_filled_components=None)
    #print()
    datamodule = clf(name=data_option,
                     in_components=['accX', 'accY', 'accZ'], out_components=['PCE'], 
                     in_chunk=[-25, 25], out_chunk=[0, 0], 
                     split_portions=(0.6, 0.2, 0.2), 
                     seed=333, batch_size=32, limit=10000, 
                     min_slice=10, scaling_method='z-norm', 
                     scaling_tag='in_only', split_method='chronological')
    
    #trainer_loader = datamodule.get_dataloader('train')
    
    #print(trainer_loader)
    
    #penguin_pickle = open("/home/randle/Dev/KnowIt/datasets/dummy_zero.pickle", "rb")
    #penguin_dict = pickle.load(penguin_pickle)
    
    #print(penguin_dict.keys())
    #print(penguin_dict['the_data'].attr)


main()