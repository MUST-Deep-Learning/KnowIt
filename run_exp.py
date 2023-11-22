from setup.ki_setup import KI_setup

action = 'train'
device = 'gpu'
experiment_name = 'default_experiment'
safe_mode = True

setup_module = KI_setup(action, experiment_name, device, safe_mode)

ping = 0