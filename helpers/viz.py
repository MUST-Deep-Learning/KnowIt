import os

from env.env_paths import (learning_data_path, learning_curves_path,
                           ckpt_path, model_args_path, model_predictions_dir,
                           model_output_dir, model_interpretations_dir)
from helpers.read_configs import load_from_csv, yaml_to_dict, load_from_path, safe_mkdir
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta, datetime
import pytz
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.dates as mdates
import glob

from helpers.logger import get_logger

logger = get_logger()

train_color = 'rebeccapurple'
valid_color = 'deeppink'
eval_color = 'crimson'
back_color = 'black'
grid_color = 'dimgray'
generic_figsize = (10, 6)
large_figsize = (20, 11.25)
generic_dpi = 200
quick_dpi = 100
generic_cmap = 'plasma'
c_cycle = [train_color, valid_color, eval_color, 'darkorange', 'mediumblue']
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=c_cycle)


def feature_attribution(id_args, interpret_args):

    interpretation_dir = model_interpretations_dir(id_args['experiment_name'], id_args['model_name'])
    predictions_dir = model_predictions_dir(id_args['experiment_name'], id_args['model_name'])

    model_args = yaml_to_dict(model_args_path(id_args['experiment_name'],
                                              id_args['model_name']))

    file_name = '_' + interpret_args['interpretation_set'] + '-' + 'ist_inx_dict' + '.pickle'
    ist_values, _ = load_from_path(os.path.join(predictions_dir, file_name))

    file_name = ''
    for a in interpret_args:
        file_name += str(interpret_args[a]) + '-'
    for f in os.listdir(interpretation_dir):
        if file_name in f and '.pickle' in f:
            file_name = f
            break
    file_path = os.path.join(interpretation_dir, file_name)
    feat_att_dict = load_from_path(file_path)

    feat_att = feat_att_dict['results']
    relevant_ist = {}
    for x in range(feat_att_dict['i_inx'][0], feat_att_dict['i_inx'][1]):
        relevant_ist[x] = ist_values[x]

    # wanted to incorporate input and output features to vizuals, but difficult and not enough time.
    # feature_values = fetch_feature_values(predictions_dir,
    #                                       interpret_args['interpretation_set'],
    #                                       relevant_ist, model_args)

    folder_name = file_name.split('.pickle')[0]
    save_dir = os.path.join(interpretation_dir, folder_name)
    safe_mkdir(save_dir, True, False)

    if model_args['data']['task'] == 'regression':

        # plot_MD_sanity_check_1(feat_att, relevant_ist, model_args, save_dir, interpret_args)
        # plot_MD_sanity_check_2(feat_att, fetch_feature_values(predictions_dir,
        #                                                       interpret_args['interpretation_set'], relevant_ist, model_args))

        plot_mean_feat_att_regression(feat_att, relevant_ist, model_args, save_dir, interpret_args)
        plot_feat_att_regression(feat_att, relevant_ist, model_args, save_dir, interpret_args)
    elif model_args['data']['task'] == 'classification':
        plot_mean_feat_att_regression(feat_att, relevant_ist, model_args, save_dir, interpret_args)
        plot_feat_att_classification(feat_att, relevant_ist, model_args, save_dir, interpret_args)


def plot_MD_sanity_check_2(feat_att, output_vals):

    order = np.array(list(feat_att.keys()))
    order = np.sort(order)

    new_ov = {}
    for o in order:
        new_ov[o] = output_vals[o]
    output_vals = new_ov


    full_fa = defaultdict(list)
    deltas = defaultdict(list)

    for pp in feat_att:
        output_logits = list(feat_att[pp].keys())
        for logit in output_logits:
            full_fa[logit].append(feat_att[pp][logit]['attributions'].detach().numpy())
            deltas[logit].append(feat_att[pp][logit]['delta'].detach().numpy())
    full_fa = dict(full_fa)
    deltas = dict(deltas)

    logits = list(full_fa.keys())


    y = [output_vals[t][0].item() for t in output_vals]
    y_hat = [output_vals[t][1].item() for t in output_vals]

    for logit in logits:
        full_fa[logit] = np.array(full_fa[logit])
        deltas[logit] = np.array(deltas[logit])
        sum_of_fa = np.sum(full_fa[logit], axis=1)
        sum_of_fa = np.sum(sum_of_fa, axis=1)
        mean_of_deltas = np.mean(deltas[logit], axis=1)

        plt.plot(sum_of_fa, label='sum of attributions')
        plt.plot(y, label='true y')
        plt.plot(y_hat, label='predicted y')
        plt.plot(mean_of_deltas, label='mean of deltas (across 1000 baselines)')
        plt.plot(sum_of_fa - mean_of_deltas, label='sum of attributions - mean of deltas')

        plt.legend()
        plt.xlabel('Prediction point time')
        plt.ylabel('Feature attribution')
        plt.title('y(t) = x1 + 2*x2(t-2) + 0.5*x3(x-4)')
        plt.show()
        plt.close()

    exit(101)


def plot_MD_sanity_check_1(feat_att, relevant_ist, model_args, save_dir, interpret_args):

    full_fa = defaultdict(list)

    for pp in feat_att:
        output_logits = list(feat_att[pp].keys())
        for logit in output_logits:
            full_fa[logit].append(feat_att[pp][logit]['attributions'].detach().numpy())
    full_fa = dict(full_fa)

    logits = list(full_fa.keys())

    # (x1, t=0), (x2, t=-2), (x3=-4)
    relevant_x_spec = [[0, 5], [1, 3], [2, 1]]

    relevant_x = []
    for x in range(0, 4):
        for t in range(0, 11):
            relevant_x.append([x, t])

    for logit in logits:
        full_fa[logit] = np.array(full_fa[logit])
        for x in relevant_x:
            relevant_fa = full_fa[logit][:, x[1], x[0]]
            if x in relevant_x_spec:
                plt.plot(relevant_fa, label='x' + str(x[0] + 1) + ', t=' + str(x[1] - 5), marker='*')
            else:
                plt.plot(relevant_fa, label='x' + str(x[0]+1) + ', t=' + str(x[1]-5))
        plt.legend(ncol=2)
        plt.xlabel('Prediction point time')
        plt.ylabel('Feature attribution')
        plt.title('y(t) = x1 + 2*x2(t-2) + 0.5*x3(x-4)')
        plt.show()
        plt.close()

    # exit(101)


def plot_mean_feat_att_regression(feat_att, relevant_ist, model_args, save_dir, interpret_args):

    full_fa = defaultdict(list)

    for pp in feat_att:
        output_logits = list(feat_att[pp].keys())
        for logit in output_logits:
            full_fa[logit].append(feat_att[pp][logit]['attributions'].detach().numpy())
    full_fa = dict(full_fa)

    logits = list(full_fa.keys())

    mean_over_pp = {}
    mean_over_time = {}
    mean_over_components = {}

    for logit in logits:
        full_fa[logit] = np.abs(full_fa[logit])
        full_fa[logit] = np.stack(full_fa[logit], axis=0)
        mean_over_pp[logit] = np.mean(full_fa[logit], axis=0)
        mean_over_time[logit] = np.mean(full_fa[logit], axis=1)
        mean_over_components[logit] = np.mean(full_fa[logit], axis=2)

    in_chunk = np.arange(model_args['data']['in_chunk'][0], model_args['data']['in_chunk'][1]+1)
    in_components = model_args['data']['in_components']

    for logit in logits:
        fig, axes = plt.subplots(1, 3, figsize=large_figsize)

        ax = axes[0]
        im = ax.imshow(mean_over_pp[logit], cmap=generic_cmap, aspect='auto')
        ax.set_ylabel('(Input) Time')
        ax.set_yticks([x for x in range(len(in_chunk))])
        ax.set_yticklabels(in_chunk)
        ax.set_xlabel('Components')
        ax.set_xticks([x for x in range(len(in_components))])
        ax.set_xticklabels(in_components)
        # ax.set_title('Mean over prediction points')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        ax = axes[1]
        im = ax.imshow(mean_over_time[logit], cmap=generic_cmap, aspect='auto')
        ax.set_ylabel('Prediction point')
        ax.set_yticks([])
        ax.set_xlabel('(Input) Components')
        ax.set_xticks([x for x in range(len(in_components))])
        ax.set_xticklabels(in_components)
        # ax.set_title('Mean over (input) time')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        ax = axes[2]
        im = ax.imshow(mean_over_components[logit], cmap=generic_cmap, aspect='auto')
        ax.set_ylabel('Prediction point')
        ax.set_yticks([])
        ax.set_xlabel('(Input) Time')
        ax.set_xticks([x for x in range(len(in_chunk))])
        ax.set_xticklabels(in_chunk)
        # ax.set_title('Mean over (input) components')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        plt.suptitle('Mean absolute attribution plots for logit ' + str(logit))

        # plt.tight_layout()
        # plt.show()

        save_path = os.path.join(save_dir, 'mean_abs_attribution_logit_' + str(logit) + '.png')
        plt.savefig(save_path, dpi=generic_dpi)
        plt.close()


def fetch_feature_values(predictions_dir, data_tag, ist_values, model_args):




    batches = []
    for b in os.listdir(predictions_dir):
        if b.startswith(data_tag + '-' + 'batch'):
            batches.append(b)

    output_values = {}
    for b in batches:
        batch = load_from_path(os.path.join(predictions_dir, b))
        s_inx = batch[0]
        batch_pred = batch[1]
        batch_target = batch[2]
        for p in range(len(s_inx)):
            pp = s_inx[p].item()
            if pp in ist_values.keys():
                y_hat = batch_pred[p]
                y = batch_target[p]
                output_values[pp] = (y_hat, y)

    return output_values


def plot_feat_att_classification(feat_att, relevant_ist, model_args, save_dir, interpret_args):

    in_components = model_args['data']['in_components']
    out_components = list(model_args['data_dynamics']['class_set'].keys())

    in_chunk = model_args['data']['in_chunk']
    out_chunk = model_args['data']['out_chunk']

    in_time = np.arange(in_chunk[0], in_chunk[1]+1)

    prefix = interpret_args['interpretation_method'] + '-' + interpret_args['interpretation_set']

    image_paths = []
    for pp in feat_att:

        output_logits = list(feat_att[pp].keys())
        prediction_point = relevant_ist[pp]
        instance = prediction_point[0]
        time_point = prediction_point[2]


        fig, axes = plt.subplots(len(output_logits), 1, figsize=generic_figsize)
        plot_num = 0
        for logit in output_logits:

            if len(output_logits) > 1:
                ax = axes[plot_num]
            else:
                ax = axes

            fa = feat_att[pp][logit]['attributions'].detach().numpy()
            fa = fa.transpose()
            im = ax.imshow(fa, aspect='auto', cmap=generic_cmap)
            # axes[plot_num].set_title(out_components[plot_num])
            # ax.set_title('Output logit: ' + str(logit) + ' / output component: ' + str(out_components[logit[1]]))
            ax.set_title('Predicting ' + str(out_components[logit]) + ' at time position ' + str(out_chunk[logit]))
            ax.set_yticks([t for t in range(len(in_components))])
            ax.set_yticklabels(in_components)

            ax.set_xticks([t for t in range(len(in_time))])
            ax.set_xticklabels(in_time)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

            # center = np.floor(fa.shape[1]/2)
            # ax.axvline(center, linestyle='--', color='black')
            # check = 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0]) + ax.get_ylim()[0]
            # ax.text(center + 0.1, check, str(prediction_point[2]), rotation=90, color='black')

            plot_num += 1

        plt.suptitle('Prediction point: [' + str(time_point) + '] \n Instance: [' + str(instance) + ']')

        # plt.show()

        save_path = os.path.join(save_dir, prefix + '-' + str(instance) + '-' + str(time_point) + '.png')
        plt.savefig(save_path, dpi=quick_dpi)
        plt.close()

        image_paths.append(save_path)

        plt.close()

    create_gif(save_dir, image_paths)


def plot_feat_att_regression(feat_att, relevant_ist, model_args, save_dir, interpret_args):

    in_components = model_args['data']['in_components']
    out_components = model_args['data']['out_components']
    in_chunk = model_args['data']['in_chunk']
    out_chunk = model_args['data']['out_chunk']

    in_time = np.arange(in_chunk[0], in_chunk[1]+1)
    out_time = np.arange(out_chunk[0], out_chunk[1] + 1)

    prefix = interpret_args['interpretation_method'] + '-' + interpret_args['interpretation_set']

    image_paths = []

    for pp in feat_att:

        output_logits = list(feat_att[pp].keys())
        prediction_point = relevant_ist[pp]
        instance = prediction_point[0]
        time_point = prediction_point[2]


        fig, axes = plt.subplots(len(output_logits), 1, figsize=generic_figsize)
        plot_num = 0
        for logit in output_logits:

            if len(output_logits) > 1:
                ax = axes[plot_num]
            else:
                ax = axes

            fa = feat_att[pp][logit]['attributions'].detach().numpy()
            fa = fa.transpose()
            im = ax.imshow(fa, aspect='auto', cmap=generic_cmap)
            # axes[plot_num].set_title(out_components[plot_num])
            # ax.set_title('Output logit: ' + str(logit) + ' / output component: ' + str(out_components[logit[1]]))
            ax.set_title('Predicting ' + str(out_components[logit[1]]) + ' at time position ' + str(out_time[logit[0]]))
            ax.set_yticks([t for t in range(len(in_components))])
            ax.set_yticklabels(in_components)

            ax.set_xticks([t for t in range(len(in_time))])
            ax.set_xticklabels(in_time)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

            # center = np.floor(fa.shape[1]/2)
            # ax.axvline(center, linestyle='--', color='black')
            # check = 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0]) + ax.get_ylim()[0]
            # ax.text(center + 0.1, check, str(prediction_point[2]), rotation=90, color='black')

            plot_num += 1

        plt.suptitle('Prediction point: [' + str(time_point) + '] \n Instance: [' + str(instance) + ']')

        # plt.show()

        save_path = os.path.join(save_dir, prefix + '-' + str(instance) + '-' + str(time_point) + '.png')
        plt.savefig(save_path, dpi=quick_dpi)
        image_paths.append(save_path)

        # plt.show()

        plt.close()

    create_gif(save_dir, image_paths)


def create_gif(save_dir, image_paths):

    def update(i):
        im.set_array(img_arr[i])
        return im,

    from PIL import Image
    import matplotlib.animation as animation

    img_arr = []
    for f in image_paths:
        image = Image.open(f)
        img_arr.append(image)

    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    im = ax.imshow(img_arr[0], animated=True)
    plt.tight_layout()
    # for i in img_arr:
    #     im = ax.imshow(i, animated=True)

    animation_fig = animation.FuncAnimation(fig, update,
                                            frames=len(img_arr),
                                            interval=200,
                                            blit=True,
                                            repeat_delay=10,
                                            repeat=True,)

    # plt.show()

    save_path = os.path.join(save_dir, "animate.gif")

    animation_fig.save(save_path, writer="pillow")

    # plt.save(save_path, animation_fig)

    plt.close()


def set_predictions(id_args, data_tag):

    model_args = yaml_to_dict(model_args_path(id_args['experiment_name'],
                                              id_args['model_name']))

    predictions_dir = model_predictions_dir(id_args['experiment_name'], id_args['model_name'])

    file_name = '_' + data_tag + '-' + 'ist_inx_dict' + '.pickle'
    ist_values, _ = load_from_path(os.path.join(predictions_dir, file_name))

    # batches = []
    # for b in os.listdir(predictions_dir):
    #     if b.startswith(data_tag + '-' + 'batch'):
    #         batches.append(b)
    #
    # predictions = defaultdict(list)
    # targets = defaultdict(list)
    # for b in batches:
    #     batch = load_from_path(os.path.join(predictions_dir, b))
    #     s_inx = batch[0]
    #     batch_pred = batch[1]
    #     batch_target = batch[2]
    #     for p in range(len(s_inx)):
    #         pp = s_inx[p].item()
    #         i = ist_values[pp][0]
    #         t = ist_values[pp][2]
    #         y_hat = batch_pred[p]
    #         y = batch_target[p]
    #         predictions[i].append([t, y_hat.numpy()])
    #         targets[i].append([t, y.numpy()])
    # predictions = dict(predictions)
    # targets = dict(targets)
    #
    # instances = targets.keys()
    # for i in instances:
    #     i_predictions = predictions[i]
    #     i_targets = targets[i]
    #     t = [x[0] for x in i_predictions]
    #     order = sorted(range(len(t)), key=t.__getitem__)
    #     t = np.array([t[x] for x in order])
    #     y = np.array([i_targets[x][1] for x in order])
    #     y_hat = np.array([i_predictions[x][1] for x in order])
    #
    #     deltas = np.diff(t)
    #     delta = deltas.min()
    #     if (deltas != delta).any():
    #         # insert NaNs in gaps
    #         a_nice_nan = np.empty_like(y[0])
    #         a_nice_nan[:] = np.nan
    #         here = np.argwhere(deltas != delta)
    #         gaps = []
    #         for h in range(len(here)):
    #             to_insert_nan = here[h].item()
    #             gap = np.arange(start=t[to_insert_nan] + delta,
    #                             stop=t[to_insert_nan + 1],
    #                             step=delta).astype(datetime)
    #             gap = np.array([pytz.utc.localize(g) for g in gap])
    #             gaps.append(gap)
    #         for g in range(len(gaps)-1, 0, -1):
    #             to_insert_nan = here[g].item() + 1
    #             t = np.insert(t, to_insert_nan, gaps[g])
    #             nan_stack = np.stack([a_nice_nan for x in range(len(gaps[g]))], axis=0)
    #             y = np.insert(y, to_insert_nan, nan_stack, axis=0)
    #             y_hat = np.insert(y_hat, to_insert_nan, nan_stack, axis=0)
    #         deltas = np.diff(t)
    #         if delta != deltas.min():
    #             logger.error('Something went very wrong with growing gaps in prediction vizuals.')
    #             exit(101)
    #
    #     predictions[i] = [t, y_hat]
    #     targets[i] = [t, y]

    instances, predictions, targets = fetch_predictions(predictions_dir, data_tag, ist_values)

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


def fetch_predictions(predictions_dir, data_tag, ist_values):

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
            for g in range(len(gaps) - 1, 0, -1):
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

    return instances, predictions, targets


def regression_set_prediction(i, predictions, targets, data_tag, out_components, predictions_dir):

    fig, ax = plt.subplots(1, 1, figsize=large_figsize)
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
    plt.close()


def classification_set_prediction(i, predictions, targets, data_tag,
                                  out_components, predictions_dir, model_args):

    def special_nanargmax(arr, axis):
        new_arr = np.zeros(arr.shape[0]) + np.nan
        d0 = np.isnan(arr).all(axis=axis)
        new_arr[~d0] = np.nanargmax(arr[~d0], axis=1)
        return new_arr

    fig, axes = plt.subplots(2, 1, figsize=large_figsize)
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

    conf_over_time_mat = np.zeros(shape=(len(class_set) * len(class_set), len(x)))
    conf_over_time_keys = [] # (predicted, target)
    row = 0
    for c_predicted in range(len(class_set)):
        for c_target in range(len(class_set)):
            t_hits = y == c_target
            p_hits = y_hat == c_predicted
            new_row = np.logical_and(t_hits, p_hits)
            new_row = 1 * new_row
            conf_over_time_mat[row, :] = new_row
            row += 1
            conf_over_time_keys.append((class_labels[c_predicted], class_labels[c_target]))


    conf_over_time_mat[:, np.isnan(y)] = np.nan

    x_lims = (x.min(), x.max())
    x_lims = mdates.date2num(x_lims)
    y_lims = [0, len(conf_over_time_keys)]

    ax = axes[0]
    # ax.set_anchor('W')
    # ax.plot(x, y, label='Target class')
    # ax.plot(x, y_hat, label='Predicted class')

    # ax.imshow(conf_over_time_mat, cmap=generic_cmap, aspect='auto', interpolation='None')

    ax.imshow(conf_over_time_mat, cmap=generic_cmap, aspect='auto',
              interpolation='None', extent=[x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], origin='bottom')
    ax.xaxis_date()

    # ax.imshow(conf_over_time_mat, cmap=generic_cmap, aspect='auto')
    # ax.set_xlim(x.min(), x.max())
    ax.set_title('(' + data_tag + ') (instance): ' + str(i) + ' (accuracy) ' + str(accuracy))
    ax.set_xlabel('Time')
    ax.set_ylabel('Class confusion (predicted, target)')
    ax.set_facecolor(back_color)
    ax.set_yticks([c+0.5 for c in range(len(conf_over_time_keys))])
    ax.set_yticklabels(conf_over_time_keys)
    # ax.grid(color=grid_color, alpha=0.5)
    # ax.legend()
    ax = axes[1]
    # ax.set_anchor('NE')
    num_classes = len(class_set)
    conf_mat = np.zeros(shape=(num_classes, num_classes))
    for c_predicted in range(num_classes):
        for c_target in range(num_classes):
            t_hits = y == c_target
            p_hits = y_hat == c_predicted
            val = np.count_nonzero(np.logical_and(t_hits, p_hits))
            conf_mat[c_target, c_predicted] = val

    ConfusionMatrixDisplay(confusion_matrix=conf_mat,
                           display_labels=[c for c in class_set.keys()]).plot(ax=ax, cmap=generic_cmap, im_kw={'aspect': 'auto'})
    plt.tight_layout()
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