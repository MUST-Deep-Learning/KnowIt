import os

from env.env_paths import (learning_data_path, learning_curves_path,
                           ckpt_path, model_args_path, model_predictions_dir, model_output_dir)
from helpers.read_configs import load_from_csv, yaml_to_dict, load_from_path
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta, datetime
import pytz
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from helpers.logger import get_logger

logger = get_logger()

train_color = 'rebeccapurple'
valid_color = 'deeppink'
eval_color = 'crimson'
back_color = 'black'
grid_color = 'dimgray'
generic_figsize = (10, 6)
generic_dpi = 200
generic_cmap = 'plasma'
c_cycle = [train_color, valid_color, eval_color, 'darkorange', 'mediumblue']
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=c_cycle)


def set_predictions(id_args, data_tag):

    model_args = yaml_to_dict(model_args_path(id_args['experiment_name'],
                                              id_args['model_name']))

    predictions_dir = model_predictions_dir(id_args['experiment_name'], id_args['model_name'])

    file_name = '_' + data_tag + '-' + 'ist_inx_dict' + '.pickle'
    ist_values, _ = load_from_path(os.path.join(predictions_dir, file_name))

    batches = []
    for b in os.listdir(predictions_dir):
        if b.startswith(data_tag + '-' + 'batch'):
            batches.append(b)

    predictions = defaultdict(list)
    targets = defaultdict(list)
    for b in batches:
        batch = load_from_path(os.path.join(predictions_dir, b))
        s_inx = batch[0]
        batch_pred = batch[1]
        batch_target = batch[2]
        for p in range(len(s_inx)):
            pp = s_inx[p].item()
            i = ist_values[pp][0]
            t = ist_values[pp][2]
            y_hat = batch_pred[p]
            y = batch_target[p]
            predictions[i].append([t, y_hat.numpy()])
            targets[i].append([t, y.numpy()])
    predictions = dict(predictions)
    targets = dict(targets)

    instances = targets.keys()
    for i in instances:
        i_predictions = predictions[i]
        i_targets = targets[i]
        t = [x[0] for x in i_predictions]
        order = sorted(range(len(t)), key=t.__getitem__)
        t = np.array([t[x] for x in order])
        y = np.array([i_targets[x][1] for x in order])
        y_hat = np.array([i_predictions[x][1] for x in order])

        deltas = np.diff(t)
        delta = deltas.min()
        if (deltas != delta).any():
            # insert NaNs in gaps
            a_nice_nan = np.empty_like(y[0])
            a_nice_nan[:] = np.nan
            here = np.argwhere(deltas != delta)
            gaps = []
            for h in range(len(here)):
                to_insert_nan = here[h].item()
                gap = np.arange(start=t[to_insert_nan] + delta,
                                stop=t[to_insert_nan + 1],
                                step=delta).astype(datetime)
                gap = np.array([pytz.utc.localize(g) for g in gap])
                gaps.append(gap)
            for g in range(len(gaps)-1, 0, -1):
                to_insert_nan = here[g].item() + 1
                t = np.insert(t, to_insert_nan, gaps[g])
                nan_stack = np.stack([a_nice_nan for x in range(len(gaps[g]))], axis=0)
                y = np.insert(y, to_insert_nan, nan_stack, axis=0)
                y_hat = np.insert(y_hat, to_insert_nan, nan_stack, axis=0)
            deltas = np.diff(t)
            if delta != deltas.min():
                logger.error('Something went very wrong with growing gaps in prediction vizuals.')
                exit(101)

        predictions[i] = [t, y_hat]
        targets[i] = [t, y]

    out_components = model_args['data']['out_components']
    out_chunk = model_args['data']['out_chunk']
    task = model_args['data']['task']

    out_range = out_chunk[1] - out_chunk[0]
    if out_range > 0 and task == 'classification':
        logger.error('Cannot predict multiple output time steps if classification.')
        exit(101)

    for i in instances:

        if model_args['data']['task'] == 'regression':
            regression_set_prediction(i, predictions, targets, data_tag,
                                      out_components, predictions_dir)
        elif model_args['data']['task'] == 'classification':
            classification_set_prediction(i, predictions, targets, data_tag,
                                          out_components, predictions_dir, model_args)
        else:
            logger.error('Unknown task type %s.', model_args['data']['task'])
            exit(101)


def regression_set_prediction(i, predictions, targets, data_tag, out_components, predictions_dir):

    fig, ax = plt.subplots(1, 1, figsize=generic_figsize)
    x = predictions[i][0]
    y_hat = predictions[i][1]
    y = targets[i][1]

    y_time = y_hat.shape[1]
    y_components = y_hat.shape[2]

    for t in range(y_time):
        for c in range(y_components):
            map = ax.plot(x, y[:, t, c],
                          label='Target ' + out_components[c] + ' step ' + str(t))
            mae = np.mean(np.abs(y[:, t, c] - y_hat[:, t, c]))
            ax.plot(x, y_hat[:, t, c],
                    label='Predicted ' + out_components[c] + ' step ' + str(t) + ' (mae=' + str(mae) + ')',
                    linestyle='--', color=map[-1].get_color())
    ax.set_title(data_tag + ' Instance: ' + str(i))
    ax.set_xlabel('Time')
    ax.set_ylabel('Feature value')
    ax.set_facecolor(back_color)
    ax.grid(color=grid_color, alpha=0.5)
    plt.legend()

    save_path = os.path.join(predictions_dir, data_tag + '-prediction-' + str(i) + '.png')
    plt.savefig(save_path, dpi=generic_dpi)


def classification_set_prediction(i, predictions, targets, data_tag,
                                  out_components, predictions_dir, model_args):

    def special_nanargmax(arr, axis):
        new_arr = np.zeros(arr.shape[0]) + np.nan
        d0 = np.isnan(arr).all(axis=axis)
        new_arr[~d0] = np.nanargmax(arr[~d0], axis=1)
        return new_arr

    fig, axes = plt.subplots(1, 2, figsize=generic_figsize)
    x = predictions[i][0]
    y_hat = special_nanargmax(predictions[i][1], axis=1)
    y = special_nanargmax(targets[i][1], axis=1)
    class_set = model_args['data_dynamics']['class_set']
    class_labels = [c for c in class_set.keys()]
    correct = y == y_hat
    correct = 1. * correct
    nan_mask = np.isnan(y)
    correct[nan_mask] = np.nan
    accuracy = np.count_nonzero(correct[~nan_mask]) / float(len(correct[~nan_mask]))
    ax = axes[0]
    ax.plot(x, y, label='Target class')
    ax.plot(x, y_hat, label='Predicted class')
    ax.set_xlim(x.min(), x.max())
    ax.set_title(data_tag + ' Instance: ' + str(i) + ': accuracy ' + str(accuracy))
    ax.set_xlabel('Time')
    ax.set_ylabel('Class')
    ax.set_facecolor(back_color)
    ax.set_yticks([c for c in range(len(class_labels))])
    ax.set_yticklabels(class_labels)
    ax.grid(color=grid_color, alpha=0.5)
    ax.legend()
    ax = axes[1]
    num_classes = len(class_set)
    conf_mat = np.zeros(shape=(num_classes, num_classes))
    for c_predicted in range(num_classes):
        for c_target in range(num_classes):
            t_hits = y == c_target
            p_hits = y_hat == c_predicted
            val = np.count_nonzero(np.logical_and(t_hits, p_hits))
            conf_mat[c_target, c_predicted] = val

    ConfusionMatrixDisplay(confusion_matrix=conf_mat,
                           display_labels=[c for c in class_set.keys()]).plot(ax=ax, cmap=generic_cmap)
    plt.tight_layout()
    save_path = os.path.join(predictions_dir, data_tag + '-prediction-' + str(i) + '.png')
    plt.savefig(save_path, dpi=generic_dpi)

def learning_curves(id_args):

    def get_curves(experiment_name, model_name):
        csv_path = learning_data_path(experiment_name, model_name)
        curve_data = load_from_csv(csv_path)
        curves = defaultdict(dict)
        for row in curve_data:
            for c in row:
                if row['epoch'] not in curves[c] or curves[c][row['epoch']] == '':
                    curves[c][row['epoch']] = row[c]
        curves = dict(curves)

        num_epochs = len(curves['epoch']) - 1
        curves.pop('step')
        curves.pop('epoch')
        # curves.pop('train_loss')

        for c in curves:
            new_curve = []
            for val in curves[c]:
                try:
                    new_curve.append(float(curves[c][val]))
                except:
                    new_curve.append(None)
            curves[c] = new_curve


        return curves, num_epochs

    def get_result_epoch(curves):
        check = []
        for x in np.array(curves['valid_loss']):
            if x is not None:
                check.append(np.abs(x - curves['result_valid_loss'][-1]))
            else:
                check.append(np.NaN)
        check = np.array(check)
        result_epoch = np.nanargmin(check)

        result_keys = [key for key in curves.keys() if 'result' in key]
        result = {}
        for r in result_keys:
            result[r] = np.round(curves[r][-1], 3)
            curves.pop(r)

        for c in curves:
            curves[c] = curves[c][:-1]

        return result, result_epoch+1

    curves, num_epochs = get_curves(id_args['experiment_name'], id_args['model_name'])
    result, result_epoch = get_result_epoch(curves)

    loss_curves = [key for key in curves.keys() if 'perf' not in key]
    perf_curves = [key for key in curves.keys() if 'perf' in key]

    epochs = [e+1 for e in range(num_epochs)]

    if len(perf_curves) > 0:
        fig, axes = plt.subplots(2, 1, figsize=generic_figsize)
        ax = axes[0]
    else:
        fig, axes = plt.subplots(1, 1, figsize=generic_figsize)
        ax = axes

    for c in loss_curves:
        ax.plot(epochs, curves[c], label=c, marker='.', color=get_color(c))
    ax.axvline(x=result_epoch, linestyle='--', c='white')
    check = 0.5 * (ax.get_ylim()[1] - ax.get_ylim()[0]) + ax.get_ylim()[0]
    ax.text(result_epoch + 0.1, check, 'model', rotation=90, color='white')
    # axes[0].set_xticks([x for x in range(num_epochs)])
    # axes[0].set_xticklabels([x + 1 for x in range(num_epochs)])

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_facecolor(back_color)
    ax.grid(color=grid_color, alpha=0.5)
    ax.legend()

    if len(perf_curves) > 0:
        ax = axes[1]
        for c in perf_curves:
            ax.plot(epochs, curves[c], label=c, marker='.', color=get_color(c))
        ax.axvline(x=result_epoch, linestyle='--', c='white')
        check = 0.5 * (ax.get_ylim()[1] - ax.get_ylim()[0]) + ax.get_ylim()[0]
        ax.text(result_epoch + 0.1, check, 'model', rotation=90, color='white')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Performance metric')
        ax.set_facecolor(back_color)
        ax.grid(color=grid_color, alpha=0.5)
        ax.legend()


    plt.suptitle(str(result), wrap=True)
    # plt.tight_layout()
    # plt.show()

    save_path = learning_curves_path(id_args['experiment_name'], id_args['model_name'])
    plt.savefig(save_path, dpi=generic_dpi)


    plt.close()


def get_color(tag):

    if 'train' in tag or 'Train' in tag:
        return train_color
    elif 'valid' in tag or 'Valid' in tag:
        return valid_color
    elif 'eval' in tag or 'Eval' in tag:
        return eval_color
    else:
        return 'yellow'