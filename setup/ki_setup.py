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


def yaml_to_dict(config_path):
    """Converts parameters in provided config yaml file to dictionary values

    Args:
        config_path (str): Path to the config file.

    Returns:
        dict: Dictionary variable containing all converted config values.

    """
    f = open(config_path, 'r')
    cfg_yaml = None
    try:
        cfg_yaml = yaml.full_load(f)
    except Exception as e:
        logger.error('Error loading config %s:\n%s' % (config_path, str(e)))
        exit(101)
    finally:
        f.close()

    cfg = ConfigDict()
    for key in cfg_yaml.keys():
        cfg[key] = cfg_yaml[key]['value']

    return cfg


class ConfigDict(dict):
    """ Adds missing key handling with default values to the builtin dict. """

    def __init__(self):
        super().__init__()

    def preprocess_get(self, key, value):
        """Convert any config values of string 'None' to type None.

        Returns:
            None: Converted None type.
        """
        if str(value).upper() == 'NONE':
            value = None
        return value

    def __getitem__(self, key):
        # If key not in dict, assign default value
        if key not in self.keys():
            if key not in cfg_default.keys():
                logging.error(f'"{key}" not found in config or default config.')
                exit(101)
            value = cfg_default[key]
            logging.warning(f'"{key}" not found in config. '
                            f'Using default: {value}')
            return self.preprocess_get(key, value)

        value = super(ConfigDict, self).__getitem__(key)
        return self.preprocess_get(key, value)
