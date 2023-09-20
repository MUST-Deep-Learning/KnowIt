__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the ki_setup module.'

from helpers.logger import get_logger
from sys import exit
import yaml

logger = get_logger()


acceptable_args = {'action': ('prepare', 'analyze', 'train', 'train-tune', 'interpret'),
                   'model': ('tcn', ),
                   'data': ('custom', ),
                   'hardware': ('cpu', 'gpu')}



class KI_setup:
    def __init__(self, action, model, data, hardware, args_dict=None):
        """init class from dictionary"""

        self.action = check_arg(action, 'action')
        self.model = check_arg(model, 'model')
        self.data = check_arg(data, 'data')
        self.hardware = check_arg(hardware, 'hardware')

        self.model_args = {}
        self.data_args = {}
        self.trainer_args = {}
        self.tuner_args = {}
        self.analyze_args = {}

    @classmethod
    def fromdict(cls, action, model, data, hardware, args_dict):
        """create class from dictionary """
        return cls(action, model, data, hardware, args_dict)

    @classmethod
    def fromyaml(cls, action, model, data, hardware, yaml_path):
        """create class from yaml """
        args_dict = yaml_to_dict(yaml_path)
        return cls(action, model, data, hardware, args_dict)



def check_arg(val, tag):
    if val in acceptable_args[tag]:
        return val
    else:
        logger.error('Unknown %s: %s', tag, val)
        exit(101)



