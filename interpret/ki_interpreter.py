__author__ = 'randlerabe@gmail.com'
__description__ = 'Contains the ki_interpreter module.'

import torch

# Todo:
# User provides s_id point (pp) or range s_id (list or tuple of pp begin/endpoints)
# User choice must fit into their choice of dataloader (train, val, eval)
# Then, extract tensors from chosen dataloader and pass into captum.
# 
# Returns: 
# We have an in_chunk of size (in_components, time_steps)
# We have an out_chunk of size (out_components, time_steps) -> for classification, this will be the classes
# We must return a results matrix for EACH POINT in the out_chunk (or each class if classification)
#
# For a single pp:
#   We choose an s_id inside the dataset underlying the chosen dataloader
#   Then extract the tensors from the dataloader. Pass to Captum.
#
# For a range pps:
#   Exactly the same as above, but many more attribution matrics since we now have a range.
#
# 

class KIInterpreter():
    """
    To fill
    """
    
    def __init__(self,
                 model,
                 model_params,
                 datamodule,
                 path_to_checkpoint):
        
        self.model = self.__load_model_from_ckpt(model=model, model_params=model_params, ckpt_path=path_to_checkpoint)    
        self.datamodule = datamodule
        
    def __load_model_from_ckpt(self, model, model_params, ckpt_path):
        
        # init Pytorch model with user params
        pt_model = model(**model_params)
        
        # obtain state of model from ckpt
        ckpt = torch.load(f=ckpt_path)
        state_dict = ckpt['state_dict']

        # todo: PL saves keys as "model.model..."; Pytorch expects "model....". Fix.
        # Approach below is temp solution
        for key in list(state_dict.keys()):
            state_dict[key[6:]] = state_dict[key]
            del state_dict[key]

        # load model and set to eval mode
        pt_model.load_state_dict(state_dict)
        pt_model.eval()
        
        return pt_model
    
        
        
        
        
        