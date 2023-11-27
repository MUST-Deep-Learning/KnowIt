__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Extracts arguments, necessary for training a model, from the experiment dictionary.'

from helpers.logger import get_logger

logger = get_logger()

required_data_args = ('data', 'in_components', 'out_components',
                      'in_chunk', 'out_chunk', 'split_portions',
                      'batch_size')
optional_data_args = ('limit', 'min_slice', 'scaling_method',
                      'scaling_tag', 'split_method', 'seed')

required_trainer_args = ('loss_fn', 'optim', 'max_epochs', 'learning_rate')
optional_trainer_args = ('learning_rate_scheduler', 'gradient_clip_val',
                         'gradient_clip_algorithm',
                         'performance_metrics', 'early_stopping', 'seed', 'return_final')


def setup_data_args(experiment_dict):

    args = {}
    for a in required_data_args:
        if a in experiment_dict.keys():
            args[a] = experiment_dict[a]
        else:
            logger.error('%s not provided in data arguments. Cannot prepare dataset.', a)
            exit(101)
    for a in optional_data_args:
        if a in experiment_dict.keys():
            args[a] = experiment_dict[a]
    args['name'] = args.pop('data')
    return args


def setup_trainer_args(experiment_dict, device, class_counts):

    args = {}
    for a in required_trainer_args:
        if a in experiment_dict.keys():
            args[a] = experiment_dict[a]
        else:
            logger.error('%s not provided in trainer arguments. Cannot create trainer.', a)
            exit(101)
    for a in optional_trainer_args:
        if a in experiment_dict.keys():
            args[a] = experiment_dict[a]

    if experiment_dict['loss_fn'] == 'weighted_cross_entropy':
        args['loss_fn'] = proc_weighted_cross_entropy(experiment_dict['task'], device, class_counts)

    return args


def proc_weighted_cross_entropy(task, device, class_counts):

    if task != 'classification' or not class_counts:
        logger.error('Weighted loss function only supported for classification tasks.')
        exit(101)

    import torch
    cc = torch.tensor([class_counts[c] for c in class_counts])
    weights = torch.sum(cc) / cc
    weights /= torch.max(weights)

    if device == 'gpu':
        weights = weights.to('cuda')

    loss_fn = {'cross_entropy': {'weight': weights}}
    return loss_fn





