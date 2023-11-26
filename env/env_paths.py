__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Environment paths.'

import os
import env.env_user as env
from helpers.logger import get_logger

logger = get_logger()


def dataset_path(option: str):
    return os.path.join(env.dataset_dir, option + '.pickle')


def arch_path(option: str):
    return os.path.join(env.archs_dir, option + '.py')


def exp_path(option: str):
    return os.path.join(env.exp_dir, option + '.yaml')


def exp_output_dir(exp_name: str):
    return os.path.join(env.project_dir, exp_name)


def model_output_dir(exp_name: str, model_name):
    return os.path.join(exp_output_dir(exp_name), model_name)


def model_args_path(exp_name: str, model_name: str):
    return os.path.join(model_output_dir(exp_name, model_name), 'model_args.yaml')


def model_interpretations_dir(exp_name: str, model_name):
    return os.path.join(model_output_dir(exp_name, model_name), 'interpretations')

def model_predictions_dir(exp_name: str, model_name):
    return os.path.join(model_output_dir(exp_name, model_name), 'predictions')


def ckpt_path(experiment_name, custom_model_name):
    # models_dir = os.path.join(os.path.join(env.project_dir, experiment_name), 'models/')
    # model_list = []
    # for m in os.listdir(models_dir):
    #     if m.startswith(custom_model_name):
    #         model_list.append(m)
    # if len(model_list) > 1:
    #     logger.warning('Found %s models with custom model name %s in %s. Selecting first.',
    #                    str(len(model_list)), custom_model_name, models_dir)
    # elif len(model_list) == 0:
    #     logger.warning('Found zero models with custom model name %s in %s.',
    #                    custom_model_name, models_dir)
    #     exit(101)
    # path = os.path.join(models_dir, model_list[0])

    path = model_output_dir(experiment_name, custom_model_name)

    ckpt_list = []
    for c in os.listdir(path):
        if c.endswith('.ckpt'):
            ckpt_list.append(c)
    if len(ckpt_list) > 1:
        logger.warning('Found %s checkpoints at %s. Selecting first.',
                       str(len(ckpt_list)), path)
    elif len(ckpt_list) == 0:
        logger.warning('Found zero checkpoints at %s.', path)
        exit(101)
    return os.path.join(path, ckpt_list[0])


def learning_data_path(exp_name: str, model_name: str):
    path = model_output_dir(exp_name, model_name)
    path = os.path.join(path, 'lightning_logs')
    path = os.path.join(path, 'version_0')
    path = os.path.join(path, 'metrics.csv')
    return path

def learning_curves_path(exp_name: str, model_name: str):
    path = model_output_dir(exp_name, model_name)
    path = os.path.join(path, 'learning_curves.png')
    return path


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    pass

# -----------------------------------------------------------------------------