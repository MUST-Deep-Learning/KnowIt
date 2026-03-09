"""
---------------
PreparedDataset
---------------

Drop-in optimized version of the KnowIt prepared dataset module.

Key optimizations:
- Avoid mutating self.batch_sampling_mode in get_dataloader()
- Support DataLoader batch fetching via __getitems__()
- Cache per-batch slice loads to avoid repeated I/O / pandas slicing
- Preload NumPy arrays directly when preload=True
- Better DataLoader defaults: pin_memory, persistent_workers, prefetch_factor
"""

from __future__ import annotations
__copyright__ = 'Copyright (c) 2025 North-West University (NWU), South Africa.'
__licence__ = 'Apache 2.0; see LICENSE file for details.'
__author__ = 'tiantheunissen@gmail.com, optimized by ChatGPT'
__description__ = ('Contains the PreparedDataset, CustomSampler, CustomDataset, and '
                   'CustomClassificationDataset, and CustomVariableLengthRegressionDataset '
                   'class for KnowIt.')

from numpy import (
    array, random, unique, pad, isnan, arange, expand_dims, concatenate,
    diff, where, split, ndarray, isscalar, asarray
)
from numpy.random import Generator
from pandas import isna, DataFrame
from torch.utils.data import Dataset, DataLoader, Sampler
from torch import from_numpy, is_tensor, Tensor, unsqueeze
from torch import zeros as zeros_tensor

from data.base_dataset import BaseDataset
from data.data_splitting import DataSplitter
from data.data_scaling import DataScaler
from helpers.logger import get_logger

logger = get_logger()


class PreparedDataset(BaseDataset):
    # to be provided
    in_components = None
    out_components = None
    in_chunk = None
    out_chunk = None
    split_portions = None
    seed = None
    batch_size = None
    split_method = None
    scaling_method = None
    scaling_tag = None
    shuffle_train = None
    limit = None
    padding_method = None
    min_slice = None
    batch_sampling_mode = None
    slide_stride = None
    variable_sequence_length_limit = None

    # to be filled automatically
    x_map = None
    y_map = None
    train_set_size = None
    valid_set_size = None
    eval_set_size = None
    selection = None
    x_scaler = None
    y_scaler = None
    in_shape = None
    out_shape = None
    class_set = None
    class_counts = None
    custom_splits = None

    def __init__(self, **kwargs) -> None:
        logger.info('Initializing PreparedClass for %s', kwargs['name'])

        super().__init__(kwargs['meta_path'], kwargs['package_path'])

        self.in_components = kwargs['in_components']
        self.out_components = kwargs['out_components']
        self.in_chunk = kwargs['in_chunk']
        self.out_chunk = kwargs['out_chunk']
        self.split_portions = kwargs['split_portions']
        self.seed = kwargs['seed']
        self.batch_size = kwargs['batch_size']
        self.split_method = kwargs['split_method']
        self.scaling_method = kwargs['scaling_method']
        self.scaling_tag = kwargs['scaling_tag']
        self.shuffle_train = kwargs['shuffle_train']
        self.limit = kwargs['limit']
        self.padding_method = kwargs['padding_method']
        self.min_slice = kwargs['min_slice']
        self.task = kwargs['task']
        self.batch_sampling_mode = kwargs['batch_sampling_mode']
        self.slide_stride = kwargs['slide_stride']
        self.variable_sequence_length_limit = kwargs['variable_sequence_length_limit']

        random.seed(self.seed)
        self._prepare()

    def get_dataset(self, set_tag: str, preload: bool = False):
        if self.task == 'regression':
            if self.batch_sampling_mode in ('variable_length', 'variable_length_inference'):
                logger.error(
                    'Batch sampling mode %s is not supported for regression tasks. '
                    'Did you mean to perform variable length regression (vl_regression)?',
                    str(self.batch_sampling_mode)
                )
                exit(101)
            dataset = CustomDataset(
                self.get_extractor(), self.selection[set_tag],
                self.x_map, self.y_map,
                self.x_scaler, self.y_scaler,
                self.in_chunk, self.out_chunk,
                self.padding_method, preload=preload
            )
        elif self.task == 'classification':
            if self.batch_sampling_mode in ('variable_length', 'variable_length_inference'):
                logger.error(
                    'Batch sampling mode %s is not supported for classification tasks.',
                    str(self.batch_sampling_mode)
                )
                exit(101)
            dataset = CustomClassificationDataset(
                self.get_extractor(), self.selection[set_tag],
                self.x_map, self.y_map,
                self.x_scaler, self.y_scaler,
                self.in_chunk, self.out_chunk,
                self.class_set,
                self.padding_method,
                preload=preload
            )
        elif self.task == 'vl_regression':
            dataset = CustomVariableLengthRegressionDataset(
                self.get_extractor(), self.selection[set_tag],
                self.x_map, self.y_map,
                self.x_scaler, self.y_scaler,
                self.in_chunk, self.out_chunk,
                self.padding_method, preload=preload
            )
        else:
            logger.error('Unknown task: %s', self.task)
            exit(101)

        return dataset

    def get_dataloader(
        self,
        set_tag: str,
        analysis: bool = False,
        num_workers: int = 4,
        preload: bool = False,
        pin_memory: bool = True,
        persistent_workers: bool | None = None,
        prefetch_factor: int = 4,
    ) -> DataLoader:
        if self.task == 'vl_regression' and self.batch_sampling_mode not in ('variable_length', 'variable_length_inference'):
            logger.error(
                'batch_sampling_mode=%s is not supported for variable length tasks. '
                'Please ensure batch_sampling_mode=variable_length if task=vl_regression.',
                str(self.batch_sampling_mode)
            )
            exit(101)

        if set_tag == 'train' and not analysis:
            shuffle = self.shuffle_train
            drop_small = True
            sampler_mode = self.batch_sampling_mode
        else:
            shuffle = False
            drop_small = False
            if self.batch_sampling_mode == 'variable_length':
                sampler_mode = 'variable_length_inference'
            else:
                sampler_mode = 'inference'

        sampler = CustomSampler(
            selection=self.selection[set_tag],
            batch_size=self.batch_size,
            input_size=self.in_shape[0],
            seed=self.seed,
            mode=sampler_mode,
            drop_small=drop_small,
            shuffle=shuffle,
            slide_stride=self.slide_stride,
            variable_sequence_length_limit=self.variable_sequence_length_limit
        )

        dataset = self.get_dataset(set_tag, preload=preload)

        if persistent_workers is None:
            persistent_workers = num_workers > 0

        dl_kwargs = {
            "dataset": dataset,
            "batch_sampler": sampler,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers if num_workers > 0 else False,
        }
        if num_workers > 0:
            dl_kwargs["prefetch_factor"] = prefetch_factor

        dataloader = DataLoader(**dl_kwargs)
        return dataloader

    def get_ist_values(self, set_tag: str) -> list:
        the_data = self.get_extractor()
        ist_values = []
        if set_tag == 'all':
            tags = ['train', 'valid', 'eval']
        elif set_tag in ['train', 'valid', 'eval']:
            tags = [set_tag]
        else:
            logger.error('Unknown set_tag {}'.format(set_tag))
            exit(101)

        for tag in tags:
            for p in self.selection[tag]:
                i = p[0]
                s = p[1]
                t_step = the_data.time_step(p[0], p[1], p[2])
                t = t_step.name
                ist_values.append((i, s, t))

        return ist_values

    def fetch_input_points_manually(self, set_tag: str, point_ids: int | list) -> dict:
        dataset = self.get_dataset(set_tag, preload=False)

        try:
            custom_batch = dataset.__getitem__(idx=point_ids)
            if len(custom_batch['x'].shape) < 3:
                custom_batch['x'] = unsqueeze(custom_batch['x'].contiguous(), 0)
                custom_batch['y'] = unsqueeze(custom_batch['y'].contiguous(), 0)
        except ValueError:
            logger.error(
                'Invalid: ids %s not in choice "%s" (which has range %s)',
                str(point_ids), set_tag, str((0, len(self.selection[set_tag])))
            )
            exit(101)

        return custom_batch

    def _prepare(self) -> None:
        missing_in_components = set(self.in_components) - set(self.components)
        if len(missing_in_components) > 0:
            logger.error('Defined in_components %s not in data option.', str(missing_in_components))
            exit(101)

        missing_out_components = set(self.out_components) - set(self.components)
        if len(missing_out_components) > 0:
            logger.error('Defined out_components %s not in data option.', str(missing_out_components))
            exit(101)

        self.y_map = array([i for i, e in enumerate(self.components) if e in self.out_components])
        self.x_map = array([i for i, e in enumerate(self.components) if e in self.in_components])

        self.in_shape = [self.in_chunk[1] - self.in_chunk[0] + 1, len(self.in_components)]
        self.out_shape = [self.out_chunk[1] - self.out_chunk[0] + 1, len(self.out_components)]

        if self.task == 'classification':
            if self.out_chunk[0] != self.out_chunk[1]:
                logger.error(
                    'Currently, KnowIt can only perform classification at one specific time step at a time. '
                    'Please change the out_chunk %s argument to reflect this. Both values must match.',
                    str(self.out_chunk)
                )
                exit(101)
            self._get_classes()
            self._count_classes()
            self.out_shape = [1, len(self.class_set)]

        logger.info('Preparing data splits (selection).')
        self.selection = DataSplitter(
            self.get_extractor(),
            self.split_method,
            self.split_portions,
            self.limit, self.y_map,
            self.in_chunk, self.out_chunk,
            self.min_slice,
            custom_splits=self.custom_splits
        ).get_selection()

        self.train_set_size = len(self.selection['train'])
        self.valid_set_size = len(self.selection['valid'])
        self.eval_set_size = len(self.selection['eval'])

        logger.info('Data split sizes: %s', str((self.train_set_size, self.valid_set_size, self.eval_set_size)))

        min_split = min(self.train_set_size, self.valid_set_size, self.eval_set_size)
        if self.batch_size > min_split:
            logger.warning(
                'Selected batch_size %s, is larger than one of dataset splits. Setting batch_size to %s.',
                str(self.batch_size), str(min_split)
            )
            self.batch_size = min_split

        logger.info('Preparing data scalers, if relevant.')

        if self.task == 'classification' and self.scaling_tag == 'full':
            logger.warning(
                'scaling_tag cannot be full for classification tasks. Changing to scaling_tag=in_only.'
            )
            self.scaling_tag = 'in_only'

        self.x_scaler, self.y_scaler = DataScaler(
            self.get_extractor(),
            self.selection['train'],
            self.scaling_method,
            self.scaling_tag,
            self.x_map,
            self.y_map
        ).get_scalers()

    def _get_classes(self) -> None:
        if self.task != 'classification':
            logger.error('Task must be classification to determine classes.')
            exit(101)

        data_extractor = self.get_extractor()
        found_class_set = set()

        for i in data_extractor.data_structure:
            vals = data_extractor.instance(i)
            subset = vals.iloc[:, self.y_map]
            mask = ~isna(subset).any(axis=1)
            vals = subset[mask]
            unique_entries = DataFrame(vals).drop_duplicates().to_numpy()
            unique_entries_list = []
            for u in unique_entries:
                if len(u) > 1:
                    unique_entries_list.append(tuple(u))
                else:
                    unique_entries_list.append(u.item())
            found_class_set.update(set(unique_entries_list))

        self.class_set = {}
        tick = 0
        for c in found_class_set:
            self.class_set[c] = tick
            tick += 1

        logger.info('Found %s unique classes.', str(len(self.class_set)))
        logger.info(self.class_set)

    def _count_classes(self) -> None:
        if self.task != 'classification' or not hasattr(self, 'class_set'):
            logger.error('Task must be classification and classes must have been determined to count classes.')
            exit(101)

        data_extractor = self.get_extractor()
        self.class_counts = {}
        for c in self.class_set:
            class_count = 0
            for i in data_extractor.data_structure:
                instance = data_extractor.instance(i).iloc[:, self.y_map]
                if isscalar(c):
                    matches = instance.eq(c)
                else:
                    c_tuple = tuple(c)
                    matches = instance.eq(c_tuple).all(axis=1)
                matches = matches.fillna(False)
                class_count += int(matches.sum())

            self.class_counts[self.class_set[c]] = class_count


class CustomSampler(Sampler):
    def __init__(
        self,
        selection: array,
        batch_size: int,
        input_size: int,
        seed: int = None,
        mode: str = 'independent',
        drop_small: bool = True,
        shuffle: bool = True,
        slide_stride: int = 1,
        variable_sequence_length_limit: int = None
    ) -> None:
        self.selection = selection
        self.batch_size = batch_size
        self.input_size = input_size
        self.seed = seed
        self.mode = mode
        self.drop_small = drop_small
        self.shuffle = shuffle
        self.slide_stride = slide_stride
        self.variable_sequence_length_limit = variable_sequence_length_limit

        self.batches = []
        self.epoch = -1
        self.set_epoch(0)

    def __iter__(self):
        if self.batches == []:
            self.set_epoch(self.epoch)
        return iter(self.batches)

    def __len__(self):
        if self.batches == []:
            self.set_epoch(self.epoch)
        return len(self.batches)

    def set_epoch(self, epoch: int):
        if epoch != self.epoch:
            self.epoch = epoch
            if self.mode == 'independent':
                self._create_default_batches()
            elif self.mode == 'sliding-window':
                self._create_sliding_window_batches()
            elif self.mode == 'inference':
                self._create_inference_batches()
            elif self.mode == 'variable_length':
                self._create_vl_batches()
            elif self.mode == 'variable_length_inference':
                self._create_vl_inference_batches()
            else:
                logger.error(
                    'Unknown sampler mode %s. Expected (independent, sliding-window, inference, '
                    'variable_length, or variable_length_inference).',
                    self.mode
                )
                exit(101)

            self._check_small()

    def _create_default_batches(self) -> None:
        self.batches = []
        sampling = arange(len(self.selection))
        if self.shuffle:
            rng = self._get_rng()
            rng.shuffle(sampling)
        for b in range(0, len(self.selection), self.batch_size):
            batch = [t for t in sampling[b:min(b + self.batch_size, len(self.selection))]]
            self.batches.append(batch)

    def _create_inference_batches(self) -> None:
        self.batches = []
        contiguous_slices = self._get_contiguous_slices()

        if self.slide_stride > 1:
            logger.warning(
                'Note: with sliding window stride greater than 1, inference is not done on all available prediction points.'
            )
            contiguous_slices = [s[::self.slide_stride] for s in contiguous_slices]

        self._block_sample_contiguous_batches(contiguous_slices)

    def _create_sliding_window_batches(self) -> None:
        self.batches = []

        rng = None
        if self.shuffle:
            rng = self._get_rng()

        contiguous_slices = self._get_contiguous_slices()

        if self.shuffle:
            rng.shuffle(contiguous_slices)

        contiguous_slices = self._expand_contiguous_slices(contiguous_slices)

        if self.shuffle:
            rng.shuffle(contiguous_slices)

        if self.shuffle:
            contiguous_slices = self._random_drop_start(contiguous_slices, rng)

        if self.slide_stride > 1:
            contiguous_slices = [s[::self.slide_stride] for s in contiguous_slices]

        self._block_sample_contiguous_batches(contiguous_slices)

    def _create_vl_batches(self) -> None:
        self.batches = []

        rng = None
        if self.shuffle:
            rng = self._get_rng()

        contiguous_slices = self._get_contiguous_slices()

        if self.shuffle:
            rng.shuffle(contiguous_slices)

        contiguous_slices = self._expand_contiguous_slices(contiguous_slices)

        if self.shuffle:
            rng.shuffle(contiguous_slices)

        if self.shuffle:
            contiguous_slices = self._random_drop_start(contiguous_slices, rng)

        self.batches = self._compile_vl_batches(contiguous_slices)

    def _create_vl_inference_batches(self) -> None:
        self.batches = []
        contiguous_slices = self._get_contiguous_slices()
        contiguous_slices.sort(key=len, reverse=True)
        self.batches = self._compile_vl_batches(contiguous_slices)

    def _compile_vl_batches(self, slices: list) -> list:
        def _fill_empty(batch, to_compile):
            new_batch = []
            for seq in batch:
                if len(seq) == 0:
                    if to_compile:
                        new_batch.append(to_compile.pop(0))
                else:
                    new_batch.append(seq)
            return new_batch, to_compile

        batches = []
        to_compile = slices.copy()

        while to_compile:
            candidate_batch, to_compile = _fill_empty(to_compile[:self.batch_size], to_compile[self.batch_size:])
            if candidate_batch:
                if self.variable_sequence_length_limit is not None:
                    min_len = min(min(len(s), self.variable_sequence_length_limit) for s in candidate_batch)
                else:
                    min_len = min(len(s) for s in candidate_batch)
                new_batch = [s[:min_len] for s in candidate_batch]
                rest_batch = [s[min_len:] for s in candidate_batch]
                batches.append(new_batch)
                to_compile = rest_batch + to_compile

        return batches

    def _block_sample_contiguous_batches(self, contiguous_slices: list) -> None:
        def _sample_new_candidate_block(slices: list, idx: int) -> list:
            return slices[idx:idx + self.batch_size]

        def _concat_slices(a: list, b: list, c: int) -> list:
            return [concatenate((a[s], b[s])) for s in range(c)]

        def _combine_with_remainder(block: list, rem_block: list) -> list:
            if rem_block is not None:
                if len(block) == 0:
                    block = rem_block
                elif len(block) == len(rem_block):
                    block = _concat_slices(rem_block, block, len(block))
                elif len(block) != len(rem_block):
                    short = min(len(block), len(rem_block))
                    long = max(len(block), len(rem_block))
                    combined = _concat_slices(rem_block, block, short)
                    if len(block) < len(rem_block):
                        combined.extend([rem_block[b] for b in range(short, long)])
                    else:
                        combined.extend([block[b] for b in range(short, long)])
                    block = combined
            return block

        def _get_min_len(block: list) -> int:
            return min([len(s) for s in block])

        def _handle_zero_slices(block: list) -> list:
            if _get_min_len(block) == 0:
                missing_entries = [_ for _ in range(len(block)) if len(block[_]) == 0]
                while len(missing_entries) > 0:
                    if missing_entries[-1] == len(block) - 1:
                        block.pop()
                        missing_entries.pop()
                    else:
                        block[missing_entries[0]] = block.pop()
                        missing_entries.pop(0)
            return block

        remainder = None
        next_block_at = 0
        while remainder != []:
            new_block = _sample_new_candidate_block(contiguous_slices, next_block_at)
            new_block = _combine_with_remainder(new_block, remainder)
            new_block = _handle_zero_slices(new_block)
            if len(new_block) == 0:
                break
            min_len = _get_min_len(new_block)
            for l in range(min_len):
                new_batch = [b[l] for b in new_block[:self.batch_size]]
                self.batches.append(new_batch)
            remainder = []
            for b in new_block[:self.batch_size]:
                remainder.append(b[min_len:])
            for b in new_block[self.batch_size:]:
                remainder.append(b)
            next_block_at += self.batch_size

    def _random_drop_start(self, contiguous_slices: list, rng: Generator, drop_max: int = 10) -> list:
        for s in range(len(contiguous_slices)):
            to_drop = rng.integers(0, min(len(contiguous_slices[s]), drop_max))
            contiguous_slices[s] = contiguous_slices[s][to_drop:]
        return contiguous_slices

    def _expand_contiguous_slices(self, contiguous_slices: list, dilation: int = 1) -> list:
        if len(contiguous_slices) == 0:
            logger.error('No contiguous slices to expand.')
            exit(101)

        if len(contiguous_slices) < self.batch_size:
            tick = 0
            found = False
            while len(contiguous_slices) < self.batch_size:
                if tick >= len(contiguous_slices):
                    tick = 0
                    if not found:
                        break
                    found = False
                if dilation < len(contiguous_slices[tick]):
                    new_slice = contiguous_slices[tick][dilation:]
                    contiguous_slices.append(new_slice)
                    found = True
                tick += 1

            if len(contiguous_slices) < self.batch_size:
                logger.error(
                    'Cannot construct ordered batches with current dataset, batch size %s and sliding-window-dilation %d',
                    self.batch_size, dilation
                )
                exit(101)

        return contiguous_slices

    def _get_contiguous_slices(self) -> list:
        inx = expand_dims(arange(len(self.selection)), 1)
        selection = concatenate((self.selection, inx), axis=1)

        instance_slice_pairs = unique(selection[:, :2], axis=0)

        contiguous_slices = []
        for s in instance_slice_pairs:
            idx = where((selection[:, 0] == s[0]) & (selection[:, 1] == s[1]))[0]
            s_block = selection[idx]
            s_block = s_block[s_block[:, 2].argsort()]
            breakpoints = where(diff(s_block[:, 2]) != 1)[0] + 1
            s_block = split(s_block, breakpoints)
            s_block = [t[:, 3] for t in s_block]
            contiguous_slices.extend(s_block)

        return contiguous_slices

    def _check_small(self) -> None:
        total_before = len(self.batches)
        if self.drop_small:
            self.batches = [b for b in self.batches if len(b) >= self.batch_size]
        total_after = len(self.batches)
        if total_before != total_after:
            logger.info(
                '%s/%s batches dropped from dataloader when dropping batches smaller than %s.',
                total_before - total_after, total_before, self.batch_size
            )

    def _get_rng(self) -> Generator:
        if self.shuffle:
            rng = random.default_rng(self.seed + self.epoch)
        else:
            rng = random.default_rng(self.seed)
        return rng

    def _batch_analyser(self) -> None:
        num_pps = len(self.selection)
        batch_sizes = [len(b) for b in self.batches]
        coverage = []
        for pp in range(num_pps):
            pp_count = 0
            batch_count = 0
            for b in self.batches:
                seen = b.count(pp)
                pp_count += seen
                if seen > 0:
                    batch_count += 1
            coverage.append([pp_count, batch_count])

        coverage = array(coverage)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 2)
        ax[0, 0].scatter([coverage[:, 0]], [coverage[:, 1]])
        ax[0, 0].set_xlabel('number of times seen')
        ax[0, 0].set_ylabel('across this many batches')

        ax[0, 1].hist(batch_sizes)
        ax[0, 1].set_xlabel('batch sizes')

        ax[1, 0].hist(coverage[:, 0])
        ax[1, 0].set_xlabel('Prediction point seen')

        ax[1, 1].hist(coverage[:, 1])
        ax[1, 1].set_xlabel('Present in this many batches')

        plt.tight_layout()
        plt.show()
        plt.close()

        exit(101)


class CustomDataset(Dataset):
    def __init__(
        self,
        data_extractor,
        selection_matrix,
        x_map,
        y_map,
        x_scaler,
        y_scaler,
        in_chunk,
        out_chunk,
        padding_method,
        preload: bool = False
    ) -> None:
        self.data_extractor = data_extractor
        self.selection_matrix = selection_matrix
        self.x_map = x_map
        self.y_map = y_map
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.in_chunk = in_chunk
        self.out_chunk = out_chunk
        self.padding_method = padding_method
        self.preload = preload

        if self.in_chunk[0] == self.in_chunk[1] and self.padding_method not in ('constant', 'empty'):
            logger.error(
                'Data padding method %s with an input chunk size of 1 not supported. '
                'Choose padding method from (constant, empty).',
                self.padding_method
            )
            exit(101)

        self.preloaded_slices = {}
        if preload:
            logger.info(
                "Preloading relevant slices into memory as NumPy arrays. "
                "This could take a while, but should speed up actual training."
            )
            instances = unique(self.selection_matrix[:, 0], axis=0)
            for i in instances:
                slices = unique(self.selection_matrix[self.selection_matrix[:, 0] == i, 1])
                for s in slices:
                    slice_vals = self.data_extractor.slice(i, s)
                    x_full = slice_vals.iloc[:, self.x_map].to_numpy(copy=True)
                    y_full = slice_vals.iloc[:, self.y_map].to_numpy(copy=True)
                    self.preloaded_slices[(i, s)] = (x_full, y_full)

    def __len__(self) -> int:
        return self.selection_matrix.shape[0]

    @staticmethod
    def _normalize_idx(idx: int | list | Tensor | ndarray) -> list:
        if type(idx) is list:
            return idx
        if is_tensor(idx):
            return idx.tolist()
        if isinstance(idx, ndarray):
            return idx.tolist()
        return [idx]

    def _get_slice_arrays(self, selection, batch_cache: dict | None = None):
        key = (selection[0], selection[1])

        if self.preload:
            return self.preloaded_slices[key]

        if batch_cache is not None and key in batch_cache:
            return batch_cache[key]

        slice_vals = self.data_extractor.slice(selection[0], selection[1])
        x_full = slice_vals.iloc[:, self.x_map].to_numpy(copy=False)
        y_full = slice_vals.iloc[:, self.y_map].to_numpy(copy=False)

        if batch_cache is not None:
            batch_cache[key] = (x_full, y_full)

        return x_full, y_full

    def _make_single_sample(self, pp: int, batch_cache: dict | None = None) -> dict:
        selection = self.selection_matrix[pp]
        x_full, y_full = self._get_slice_arrays(selection, batch_cache=batch_cache)

        x_vals = self._sample_and_pad(x_full, selection, self.in_chunk, self.padding_method)
        y_vals = y_full[selection[2] + self.out_chunk[0]: selection[2] + self.out_chunk[1] + 1, :]

        x_vals = self.x_scaler.transform(x_vals)
        y_vals = self.y_scaler.transform(y_vals)

        return self._package_output(x_vals, y_vals, pp, [selection])

    def __getitem__(self, idx: int | list | Tensor | ndarray) -> dict:
        idx_list = self._normalize_idx(idx)

        if len(idx_list) == 1:
            return self._make_single_sample(idx_list[0], batch_cache=None)

        # Manual multi-fetch path, preserving previous behaviour
        batch_cache = {}
        input_x = []
        output_y = []
        ist_idx = []

        for pp in idx_list:
            selection = self.selection_matrix[pp]
            ist_idx.append(selection)

            x_full, y_full = self._get_slice_arrays(selection, batch_cache=batch_cache)

            x_vals = self._sample_and_pad(x_full, selection, self.in_chunk, self.padding_method)
            y_vals = y_full[selection[2] + self.out_chunk[0]: selection[2] + self.out_chunk[1] + 1, :]

            input_x.append(self.x_scaler.transform(x_vals))
            output_y.append(self.y_scaler.transform(y_vals))

        input_x = array(input_x)
        output_y = array(output_y)

        return self._package_output(input_x, output_y, idx, ist_idx)

    def __getitems__(self, idx: list[int]) -> list[dict]:
        """
        Fast-path used by torch DataLoader when auto-collating batches.
        Must return a list of individual samples, not an already-collated batch.
        """
        idx_list = self._normalize_idx(idx)
        batch_cache = {}
        return [self._make_single_sample(pp, batch_cache=batch_cache) for pp in idx_list]

    @staticmethod
    def _sample_and_pad(slice_vals, selection, s_chunk, pad_mode):
        far_left = 0
        far_right = slice_vals.shape[0]
        left = selection[2] + s_chunk[0]
        right = selection[2] + s_chunk[1]

        if right < far_left or left >= far_right:
            vals = slice_vals[0:0, :]
            pad_size = s_chunk[1] - s_chunk[0] + 1
            pw = ((pad_size, 0), (0, 0))
            vals = pad(vals, pad_width=pw, mode=pad_mode)
        else:
            corrected_left = max(far_left, left)
            corrected_right = min(right, far_right)
            vals = slice_vals[corrected_left: corrected_right + 1, :]
            if left < far_left:
                pw = ((far_left - left, 0), (0, 0))
                vals = pad(vals, pad_width=pw, mode=pad_mode)
            if right >= far_right:
                pw = ((0, right - far_right + 1), (0, 0))
                vals = pad(vals, pad_width=pw, mode=pad_mode)

        return vals

    @staticmethod
    def _package_output(input_x, output_y, idx, ist_idx):
        sample = {
            'x': from_numpy(input_x).float(),
            'y': from_numpy(output_y).float(),
            's_id': idx,
            'ist_idx': ist_idx
        }
        return sample


class CustomClassificationDataset(CustomDataset):
    class_set = {}
    class_counts = {}

    def __init__(
        self,
        data_extractor,
        selection_matrix,
        x_map,
        y_map,
        x_scaler,
        y_scaler,
        in_chunk,
        out_chunk,
        class_set,
        padding_method,
        preload: bool = False
    ) -> None:
        if out_chunk[0] != out_chunk[1]:
            logger.error(
                'Currently, KnowIt can only perform classification at one specific time step at a time. '
                'Please change the out_chunk %s argument to reflect this. Both values must match.',
                str(out_chunk)
            )
            exit(101)

        super().__init__(
            data_extractor, selection_matrix, x_map, y_map,
            x_scaler, y_scaler,
            in_chunk, out_chunk,
            padding_method,
            preload
        )
        self.class_set = class_set

    def _package_output(self, input_x, output_y, idx, ist_idx):
        if len(output_y.shape) == 2:
            new_output_y = zeros_tensor(len(self.class_set))
            new_output_y[self.class_set[output_y.item()]] = 1
            output_y = new_output_y
        elif len(output_y.shape) == 3:
            new_output_y = zeros_tensor(size=(output_y.shape[0], len(self.class_set)))
            new_output_y[:, output_y] = 1
            output_y = new_output_y

        sample = {
            'x': from_numpy(input_x).float(),
            'y': output_y,
            's_id': idx,
            'ist_idx': ist_idx
        }
        return sample


class CustomVariableLengthRegressionDataset(CustomDataset):
    def __init__(
        self,
        data_extractor,
        selection_matrix,
        x_map,
        y_map,
        x_scaler,
        y_scaler,
        in_chunk,
        out_chunk,
        padding_method,
        preload: bool = False
    ) -> None:
        if in_chunk[0] != in_chunk[1]:
            logger.error('For variable length input modeling, input chunk size must be 1.')
            exit(101)

        if out_chunk[0] != out_chunk[1]:
            logger.error('For variable length input modeling, output chunk size must be 1.')
            exit(101)

        super().__init__(
            data_extractor, selection_matrix, x_map, y_map,
            x_scaler, y_scaler,
            in_chunk, out_chunk,
            padding_method,
            preload
        )

    def _package_output(self, input_x, output_y, idx, ist_idx):
        if len(input_x.shape) == 3:
            input_x = input_x.squeeze(axis=1)
            output_y = output_y.squeeze(axis=1)

        sample = {
            'x': from_numpy(input_x).float(),
            'y': from_numpy(output_y).float(),
            's_id': idx,
            'ist_idx': ist_idx
        }
        return sample