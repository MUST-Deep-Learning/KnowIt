from typing import Callable, Union, Optional

from torch.nn import functional as F
from torch import optim
from torch.optim import lr_scheduler
from torchmetrics import functional as mF
    
    
def get_loss_function(loss: str) -> Callable[[Union[int, float], Union[int, float], Optional[dict]], Union[int, float]]:
    """A helper method to retrieve the user's choice of loss function.

    Args:
        loss (str): The loss function as specified in torch.nn.functional.

    Returns:
        object: Pytorch loss function.
        
    """
    
    return getattr(F, loss)

def get_optim(optimizer: str) -> Callable[[Union[int, float], Optional[dict]], Union[int, float]]:
    """A helper method to retrieve the user's choice of optimizer.

    Args:
        optimizer (str): The loss function as specified in torch.optim.

    Returns:
        object: Pytorch optimizer function.
        
    """
    
    return getattr(optim, optimizer)

def get_lr_scheduler(scheduler: str):
    """A helper method to retrieve the user's choice of learning rate scheduler.

    Args:
        scheduler (str): The loss function as specified in torch.nn.functional.

    Returns:
        object: Pytorch learning scheduler.
        
    """
    
    return getattr(lr_scheduler, scheduler)

def get_performance_metric(metric: str) -> Callable[[Union[int, float], Union[int, float], Optional[dict]], Union[int, float]]:
    """A helper method to retrieve the user's choice of performance metric.

    Args:
        metric (str): The metric function as specified in torchmetrics.functional.

    Returns:
        object: Torchmetrics function.
        
    """
    
    return getattr(mF, metric)