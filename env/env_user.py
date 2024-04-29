__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains user-specific environment settings.'

# external imports
import os

# where to find KnowIt scripts

# repo_dir = os.path.expanduser('~/g_repos/KnowIt')
repo_dir = os.path.dirname(os.path.realpath(__file__)).split('/env')[0]

# directories containing defaults
default_dataset_dir = os.path.join(repo_dir, 'default_datasets')
default_archs_dir = os.path.join(repo_dir, 'default_archs')
temp_exp_dir = os.path.join(repo_dir, 'temp_experiments')

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    pass

# -----------------------------------------------------------------------------
