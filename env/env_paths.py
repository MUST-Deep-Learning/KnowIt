__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Environment paths.'

import os
import env.env_user as env


def dataset_path(option: str):

    return os.path.join(env.dataset_dir, option + '.pickle')

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    pass

# -----------------------------------------------------------------------------