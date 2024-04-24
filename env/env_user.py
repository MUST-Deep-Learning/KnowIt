__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains user-specific environment settings.'

# external imports
import os

# set by user

# where to find Knowit scripts
repo_dir = os.path.expanduser('~/g_repos/KnowIt')

# directories containing defaults
default_dataset_dir = os.path.join(repo_dir, 'default_datasets')
default_archs_dir = os.path.join(repo_dir, 'default_archs')
temp_exp_dir = os.path.join(repo_dir, 'experiments')

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    pass

# -----------------------------------------------------------------------------
