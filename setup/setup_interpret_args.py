__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Extracts arguments, necessary for interpreting a model, from the experiment dictionary.'

from env.env_paths import (learning_data_path, learning_curves_path,
                           ckpt_path, model_args_path, model_predictions_dir, model_output_dir)
from helpers.read_configs import load_from_csv, yaml_to_dict, load_from_path
import numpy as np
from collections import defaultdict
from datetime import timedelta, datetime
import pytz
import os
from helpers.logger import get_logger
logger = get_logger()

required_args = ('interpretation_method',)
optional_args = ('interpretation_set', 'selection', 'size',
                 'multiply_by_inputs')


def setup_interpret_args(experiment_dict):

    """ Extracts the relevant arguments for interpreting a model. """

    args = {}
    for a in required_args:
        if a in experiment_dict.keys():
            args[a] = experiment_dict[a]
        else:
            logger.error('%s not provided in interpret arguments. Cannot interpret a model.', a)
            exit(101)
    for a in optional_args:
        if a in experiment_dict.keys():
            args[a] = experiment_dict[a]
    if 'interpretation_set' not in args:
        args['interpretation_set'] = 'eval'
    if 'selection' not in args:
        args['selection'] = 'random'
    if 'size' not in args:
        args['size'] = 1
    if 'multiply_by_inputs' not in args:
        args['multiply_by_inputs'] = True
    return args


def get_interpretation_inx(interpretation_args, model_args):

    inx = 0

    if interpretation_args['interpretation_set'] == 'train':
        set_size = model_args['data_dynamics']['train_size']
    elif interpretation_args['interpretation_set'] == 'valid':
        set_size = model_args['data_dynamics']['valid_size']
    elif interpretation_args['interpretation_set'] == 'eval':
        set_size = model_args['data_dynamics']['eval_size']
    else:
        logger.error('Unknown interpretation_set %s', interpretation_args['interpretation_set'])
        exit(101)

    size = interpretation_args['size']
    if set_size < size:
        logger.error('Size of desired interpretation %s larger than desired set %s.',
                     size,
                     interpretation_args['interpretation_set'])
        exit(101)

    if interpretation_args['selection'] == 'random':
        import numpy as np
        start = np.random.randint(0, set_size - size)
        inx = (start, start + size)
    elif interpretation_args['selection'] == 'all':
        inx = (0, set_size)
    elif interpretation_args['selection'] == 'success':
        inx = select_chunk(interpretation_args, model_args, selection='success')
    elif interpretation_args['selection'] == 'failure':
        inx = select_chunk(interpretation_args, model_args, selection='failure')
    else:
        logger.error('Invalid interpretation selection %s.', interpretation_args['selection'])
        exit(101)

    return inx


def select_chunk(interpretation_args, model_args, selection):

    chunk = interpretation_args['size']

    mae = get_mae_performance(interpretation_args, model_args, selection)
    chunk_perf = np.convolve(mae, np.ones(chunk) / chunk, mode='valid')

    # import matplotlib.pyplot as plt
    # plt.plot(mae, label='mae')
    # plt.plot(chunk_perf, label='mean')
    # plt.show()
    # plt.close()

    if selection == 'success':
        select_chunk_inx = np.argmin(chunk_perf)
    else:
        select_chunk_inx = np.argmax(chunk_perf)

    select_chunk = (select_chunk_inx, select_chunk_inx + chunk)

    return select_chunk



def get_mae_performance(interpretation_args, model_args, selection):
    # TODO: This function has overlap with viz.set_predictions. Need to refactor later.

    predictions_dir = model_predictions_dir(model_args['id']['experiment_name'], model_args['id']['model_name'])

    if not os.path.exists(predictions_dir):
        logger.error('Please generate prediction values if you want to interpret %s', selection)
        exit(101)

    batches = []
    for b in os.listdir(predictions_dir):
        if b.startswith(interpretation_args['interpretation_set'] + '-' + 'batch'):
            batches.append(b)

    predictions = {}
    targets = {}
    for b in batches:
        batch = load_from_path(os.path.join(predictions_dir, b))
        s_inx = batch[0]
        for p in range(len(s_inx)):
            s = s_inx[p].item()
            y_hat = batch[1][p]
            y = batch[2][p]
            predictions[s] = y_hat.numpy()
            targets[s] = y.numpy()

    if len(predictions) != model_args['data_dynamics'][interpretation_args['interpretation_set'] + '_size']:
        logger.error('Could not find all prediction points for interpretation selection.')
        exit(101)

    points = np.array(list(predictions.keys()))
    points = np.sort(points)

    performance = []
    for p in points:
        a = predictions[p]
        b = targets[p]
        perf = np.mean(np.abs(a - b))
        performance.append(perf)

    performance = np.array(performance)


    return performance