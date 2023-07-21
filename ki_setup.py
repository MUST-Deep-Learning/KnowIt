__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the ki_setup module.'


class KI_setup:
    def __init__(self, dict):
        """init class from dictionary"""

    @classmethod
    def fromdict(cls, dict):
        """creat class from dictionary """
        return cls(dict)

    @classmethod
    def fromyaml(cls, yaml_path):
        """creat class from yaml """
        dict = yaml_to_dict(yaml_path)
        return cls(dict)
