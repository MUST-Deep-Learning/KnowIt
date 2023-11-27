import os

from env.env_paths import (learning_data_path, learning_curves_path,
                           ckpt_path, model_args_path, model_predictions_dir, model_output_dir)
from helpers.read_configs import load_from_csv, yaml_to_dict, load_from_path
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta

from helpers.logger import get_logger

logger = get_logger()

train_color = 'rebeccapurple'
valid_color = 'deeppink'
eval_color = 'crimson'
back_color = 'black'
grid_color = 'dimgray'
generic_figsize = (10, 6)
generic_dpi = 200
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

        # insert NaNs in gaps
        deltas = np.diff(t)
        delta = deltas.min()
        a_nice_nan = np.empty_like(y[0])
        a_nice_nan[:] = np.nan
        while (deltas != delta).any():
            logger.error('THIS NEEDS CHECKING ASAP!')
            exit(101)
            here = np.argwhere(deltas != delta)
            for h in range(len(here), 0, -1):
                t = np.insert(t, h, t[h] + delta)
                y = np.insert(y, h, a_nice_nan, axis=0)
                y_hat = np.insert(y_hat, h, a_nice_nan, axis=0)
            deltas = np.diff(t)
            delta = deltas.min()
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

        # plt.show()
        plt.close()


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
    else:
        fig, axes = plt.subplots(1, 1, figsize=generic_figsize)

    for c in loss_curves:
        axes[0].plot(epochs, curves[c], label=c, marker='.', color=get_color(c))
    axes[0].axvline(x=result_epoch, linestyle='--', c='white')
    check = 0.5 * (axes[0].get_ylim()[1] - axes[0].get_ylim()[0]) + axes[0].get_ylim()[0]
    axes[0].text(result_epoch + 0.1, check, 'model', rotation=90, color='white')
    # axes[0].set_xticks([x for x in range(num_epochs)])
    # axes[0].set_xticklabels([x + 1 for x in range(num_epochs)])

    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_facecolor(back_color)
    axes[0].grid(color=grid_color, alpha=0.5)
    axes[0].legend()

    if len(perf_curves) > 0:
        for c in perf_curves:
            axes[1].plot(epochs, curves[c], label=c, marker='.', color=get_color(c))
        axes[1].axvline(x=result_epoch, linestyle='--', c='white')
        check = 0.5 * (axes[1].get_ylim()[1] - axes[1].get_ylim()[0]) + axes[1].get_ylim()[0]
        axes[1].text(result_epoch + 0.1, check, 'model', rotation=90, color='white')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Performance metric')
        axes[1].set_facecolor(back_color)
        axes[1].grid(color=grid_color, alpha=0.5)
        axes[1].legend()


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