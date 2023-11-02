__author__ = 'tiantheunissen@gmail.com'
__description__ = 'User-specific environment settings.'

import os

# set by user

# where to store outputs
project_dir = os.path.expanduser('~/projects/KnowIt/')

# where to find Knowit scripts
repo_dir = os.path.expanduser('~/g_repos/KnowIt')

dataset_dir = os.path.join(repo_dir, 'datasets')
archs_dir = os.path.join(repo_dir, 'archs')
models_dir = os.path.join(project_dir, 'models')

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    pass

# -----------------------------------------------------------------------------