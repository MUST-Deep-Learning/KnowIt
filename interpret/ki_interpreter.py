__author__ = 'randlerabe@gmail.com'
__description__ = 'Contains the ki_interpreter module.'

import torch

class KIInterpreter():
    """
    To fill
    """
    
    def __init__(self,
                 model,
                 model_params,
                 datamodule,
                 i_mode,
                 i_data,
                 path_to_checkpoint):
        
        self.model = self.__load_model_from_ckpt(model=model, model_params=model_params, ckpt_path=path_to_checkpoint)
    
    def __load_model_from_ckpt(self, model, model_params, ckpt_path):
        
        # init Pytorch model with user params
        pt_model = model(**model_params)
        
        # obtain state of model from ckpt
        ckpt = torch.load(f=ckpt_path)
        state_dict = ckpt['state_dict']

        # todo: PL saves keys as "model.model..."; Pytorch expects "model....". Fix
        # approach below is temp solution
        for key in list(state_dict.keys()):
            state_dict[key[6:]] = state_dict[key]
            del state_dict[key]

        # load model and set to eval mode
        pt_model.load_state_dict(state_dict)
        pt_model.eval()
        
        return pt_model
    
    def __datamodule_something(self):
        # call the relevant dataloader from the datamodule
        # A point or range is chosen by the user.
        # We have to use the idx from the dataloader and work backwards to Knowit's data structure and extract the datapoints. This 
        # gives us the slice, instances, and components
        # 
        pass
        
        
        
        
        