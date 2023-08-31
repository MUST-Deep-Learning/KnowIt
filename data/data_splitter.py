__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the methods to split the base data in various ways.'

import numpy as np

from helpers.logger import get_logger

logger = get_logger()

def get_target_splits(data, method, portions, instances, num_targets, seed=123):

    # default seed = 123
    np.random.seed(seed)

    # time, instance, slice, relative position
    targets, times = get_target_data(data, instances)

    if len(targets) != num_targets:
        logger.error('Something went very wrong with identifying targets.')
        exit(101)

    if sum(portions) != 1.0:
        logger.error('Split portions do not add up to one.')
        exit(101)

    train_targets, valid_targets, eval_targets = do_split(targets, times, portions, method)

    return train_targets, valid_targets, eval_targets


def do_split(target_data, times, portions, method):

    if method in ('slice-random', 'slice-chronological'):
        sub_indices = split_indices_on(target_data, 'slice')
    elif method in ('instance-random', 'instance-chronological'):
        sub_indices = split_indices_on(target_data, 'instance')
    elif method in ('random', 'chronological'):
        sub_indices = split_indices_on(target_data, 'time')
    else:
        logger.error("Unknown split method %s.", method)
        exit(101)

    if method in ('slice-chronological', 'instance-chronological', 'chronological'):
        sub_starts = times[sub_indices[:, 0]]
        new_order = np.argsort(sub_starts)
    elif method in ('slice-random', 'instance-random', 'random'):
        new_order = np.arange(0, sub_indices.shape[0])
        np.random.shuffle(new_order)
    else:
        logger.error("Unknown split method %s.", method)
        exit(101)

    train_selection, valid_selection, eval_selection = ordered_split(new_order, portions)
    train_targets = sample_and_stack(target_data, sub_indices, train_selection)
    valid_targets = sample_and_stack(target_data, sub_indices, valid_selection)
    eval_targets = sample_and_stack(target_data, sub_indices, eval_selection)

    return train_targets, valid_targets, eval_targets


def sample_and_stack(data, indices, selection):
    train_targets = [data[indices[i, 0]:indices[i, 1]] for i in selection]
    train_targets = np.vstack(train_targets)
    return train_targets


def ordered_split(vals, portions):

    train_size = int(np.floor(portions[0] * len(vals)))
    eval_size = int(np.floor(portions[2] * len(vals)))
    eval_vals = vals[-eval_size:]
    train_vals = vals[:train_size]
    valid_vals = vals[train_size:-eval_size]

    return train_vals, valid_vals, eval_vals


def split_indices_on(targets, tag):

    if tag == 'instance':
        _, indices = np.unique(targets[:, 0], return_index=True)
    elif tag == 'slice':
        _, indices = np.unique(targets[:, :2], return_index=True, axis=0)
    elif tag == 'time':
        _, indices = np.unique(targets[:, :3], return_index=True, axis=0)
    else:
        logger.error("Unknown split-on tag %s.", tag)
        exit(101)

    indices = np.append(indices, targets.shape[0])

    ret_indices = []
    for i in range(len(indices)-1):
        ret_indices.append([indices[i], indices[i + 1]])
    ret_indices = np.array(ret_indices)

    return ret_indices


def get_target_data(data, instances):

    def no_nan_mask(arr):
        return ~np.isnan(arr).any(axis=1)

    def constant_col(count, val):
        return np.array([val for t in range(count)])

    def relative_col(count):
        return np.array([t for t in range(count)])

    times = []
    targets = []
    for i in range(len(instances)):
        s = 0
        for slice in data[instances[i]]:
            mask = no_nan_mask(slice['y'])
            times.append(slice['t'][mask])
            nn_count = np.count_nonzero(mask)
            targets.append(np.vstack((constant_col(nn_count, i),
                                      constant_col(nn_count, s),
                                      relative_col(nn_count))))
            s += 1

    targets = np.concatenate(targets, axis=1)
    targets = targets.transpose()
    times = np.concatenate(times)

    # targets   [instance, slice, relative position]
    # times     [timestep]

    return targets, times
