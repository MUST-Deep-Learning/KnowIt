import numpy as np
from numpy import (array, random, argwhere, isnan,
                   count_nonzero, vstack, concatenate,
                   argsort, arange, unique, append, floor)

from helpers.logger import get_logger

logger = get_logger()


class DataSplitter:

    def __init__(self, the_data, method,
                 portions, instances, limit, y_map, out_chunk, min_slice):

        self.the_data = the_data
        self.method = method
        self.portions = portions
        self.instances = instances
        self.limit = limit
        self.y_map = y_map
        self.out_chunk = out_chunk
        self.min_slice = min_slice

        if sum(self.portions) != 1.0:
            logger.error('Split portions do not add up to one.')
            exit(101)

        # [instance, slice, relative position]
        targets, times = self.__get_target_data(self.the_data, self.instances,
                                                self.y_map, self.out_chunk, self.min_slice)
        num_targets = len(times)

        if limit != -1:
            if limit > num_targets:
                logger.error('Data limiter %s larger than available training data.', str(limit))
                exit(101)
            else:
                logger.warning('Limiting data to %s points.', str(limit))
                targets = targets[:limit, :]
                times = times[:limit]

        self.train_targets, self.valid_targets, self.eval_targets = self.__do_split(targets, times,
                                                                   self.portions,
                                                                   self.method)

    def get_selection(self):
        return {'train': self.train_targets, 'valid': self.valid_targets, 'eval': self.eval_targets}

    @staticmethod
    def __sample_and_stack(data, indices, selection):
        train_targets = [data[indices[i, 0]:indices[i, 1]] for i in selection]
        train_targets = vstack(train_targets)
        return train_targets

    @staticmethod
    def __ordered_split(vals, portions):

        train_size = int(floor(portions[0] * len(vals)))
        eval_size = int(floor(portions[2] * len(vals)))
        eval_vals = vals[-eval_size:]
        train_vals = vals[:train_size]
        valid_vals = vals[train_size:-eval_size]

        return train_vals, valid_vals, eval_vals

    @staticmethod
    def __split_indices_on(targets, tag):

        if tag == 'instance':
            _, indices = unique(targets[:, 0], return_index=True)
        elif tag == 'slice':
            _, indices = unique(targets[:, :2], return_index=True, axis=0)
        elif tag == 'time':
            _, indices = unique(targets[:, :3], return_index=True, axis=0)
        else:
            logger.error("Unknown split-on tag %s.", tag)
            exit(101)

        indices = append(indices, targets.shape[0])

        ret_indices = []
        for i in range(len(indices) - 1):
            ret_indices.append([indices[i], indices[i + 1]])
        ret_indices = array(ret_indices)

        return ret_indices

    @staticmethod
    def __do_split(target_data, times, portions, method):

        if method in ('slice-random', 'slice-chronological'):
            sub_indices = DataSplitter.__split_indices_on(target_data, 'slice')
        elif method in ('instance-random', 'instance-chronological'):
            sub_indices = DataSplitter.__split_indices_on(target_data, 'instance')
        elif method in ('random', 'chronological'):
            sub_indices = DataSplitter.__split_indices_on(target_data, 'time')
        else:
            logger.error("Unknown split method %s.", method)
            exit(101)

        if method in ('slice-chronological', 'instance-chronological', 'chronological'):
            sub_starts = times[sub_indices[:, 0]]
            new_order = argsort(sub_starts)
        elif method in ('slice-random', 'instance-random', 'random'):
            new_order = arange(0, sub_indices.shape[0])
            random.shuffle(new_order)
        else:
            logger.error("Unknown split method %s.", method)
            exit(101)

        train_selection, valid_selection, eval_selection = DataSplitter.__ordered_split(new_order, portions)

        if len(train_selection) < 1 or len(valid_selection) < 1 or len(eval_selection) < 1:
            logger.error('Data segments too limited for selected splitting method %s', method)
            exit(101)

        train_targets = DataSplitter.__sample_and_stack(target_data, sub_indices, train_selection)
        valid_targets = DataSplitter.__sample_and_stack(target_data, sub_indices, valid_selection)
        eval_targets = DataSplitter.__sample_and_stack(target_data, sub_indices, eval_selection)

        return train_targets, valid_targets, eval_targets

    @staticmethod
    def __get_target_data(data, instances, y, out_chunk, min_slice):

        def no_nan_mask(arr, chunk):
            mask = ~isnan(arr).any(axis=1)
            chunk = array(chunk)
            negatives = chunk[chunk < 0]
            positives = chunk[chunk > 0]
            if len(negatives) > 0:
                mask[:abs(min(negatives))] = False
            if len(positives) > 0:
                mask[-max(positives):] = False

            return mask

        def constant_col(count, val):
            return array([val for t in range(count)])

        def relative_col(mask):
            vals = argwhere(mask)
            return vals.squeeze()

        times = []
        targets = []
        ignored_slices = 0
        for i in range(len(instances)):
            s = 0
            for slice in data[instances[i]]:
                if min_slice is None or min_slice < slice['d'].shape[0]:
                    mask = no_nan_mask(slice['d'][:, y], out_chunk)
                    times.append(slice['t'][mask])
                    nn_count = count_nonzero(mask)
                    targets.append(vstack((constant_col(nn_count, i),
                                              constant_col(nn_count, s),
                                              relative_col(mask))))
                else:
                    ignored_slices += 1
                s += 1

        if min_slice:
            logger.warning(str(ignored_slices) + ' slices ignored because they were smaller than min_slice=%s.',
                           str(min_slice))

        targets = concatenate(targets, axis=1)
        targets = targets.transpose()
        times = concatenate(times)

        # targets   [instance, slice, relative position]
        # times     [timestep]

        return targets, times