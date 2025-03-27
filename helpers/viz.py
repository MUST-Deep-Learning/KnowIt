"""This script contains example visualizations. These serve to illustrate how the outputs of KnowIt can be visualized."""

__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains various functions to visualize KnowIt data structures.'

# external imports
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.metrics import ConfusionMatrixDisplay
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import pandas as pd

# imports imports
from env.env_paths import (learning_data_path,
                           model_args_path, model_predictions_dir,
                           model_interpretations_dir, model_viz_dir, model_output_dir)
from data.base_dataset import BaseDataset
from setup.select_interpretation_points import get_predictions
from helpers.file_dir_procs import yaml_to_dict, load_from_path
from helpers.fetch_torch_mods import get_model_score
from helpers.logger import get_logger
logger = get_logger()

# ----------------------------------------------------------------------------------------------------------------------
# Global style settings
# ----------------------------------------------------------------------------------------------------------------------

plt.style.use('dark_background')

train_color = 'slategrey'
valid_color = 'rebeccapurple'
eval_color = 'crimson'

predicted_color = 'dodgerblue'
target_color = 'orangered'

grid_color = 'dimgray'
generic_figsize = (10, 6)
large_figsize = (20, 11.25)
WIDE_figsize = (40, 11.25)
TALL_figsize = (20, 24)
generic_dpi = 200
quick_dpi = 100
generic_cmap = 'plasma'
color_cycle = ['green', 'orange', 'purple', 'cyan', 'pink', 'yellow']

def get_color(tag: str) -> str:
    """ Returns a color based on a given string tag. """
    if 'train' in tag.lower():
        return train_color
    elif 'valid' in tag.lower():
        return valid_color
    elif 'eval' in tag.lower():
        return eval_color
    elif 'target' in tag.lower():
        return target_color
    elif 'predicted' in tag.lower():
        return predicted_color
    else:
        return 'yellow'

# ----------------------------------------------------------------------------------------------------------------------
# Visualize learning curves
# ----------------------------------------------------------------------------------------------------------------------

def plot_learning_curves(exp_output_dir: str, model_name: str) -> None:
    """
    Plots the learning curves for the given model based on the experiment output directory.

    Parameters
    ----------
    exp_output_dir : str
        The directory containing experiment outputs.
    model_name : str
        The name of the model for which to plot the learning curves.

    Returns
    -------
    None
    """
    curves, num_epochs = get_learning_curves(exp_output_dir, model_name)
    score, metric, result_epoch = get_model_score(model_output_dir(exp_output_dir, model_name))

    loss_curves = [key for key in curves.keys() if 'perf' not in key]
    perf_curves = [key for key in curves.keys() if 'perf' in key]
    epochs = [e + 1 for e in range(num_epochs)]
    result_epoch += 1

    num_rows = 1
    perf_set = []
    if perf_curves:
        perf_set = list(set([curve.split('_perf_')[-1] for curve in perf_curves]))
        num_rows += len(perf_set)

    fig, axes = plt.subplots(num_rows, 1, figsize=generic_figsize)
    ax = axes if isinstance(axes, np.ndarray) else [axes]

    def plot_curves(ax: plt.Axes, curves: dict, epoch: list, result_epoch: int, ylabel: str) -> None:
        """Helper function to plot curves."""
        for c in curves:
            ax.plot(epoch, curves[c], label=c, marker='.', color=get_color(c))
        ax.set_xlabel('Epochs')
        ax.set_ylabel(ylabel)
        ax.grid(color=grid_color, alpha=0.5)
        ax.legend()

    # Plot loss curves
    plot_curves(ax[0], {k: curves[k] for k in loss_curves}, epochs, result_epoch, 'Loss')
    ax[0].axvline(x=result_epoch, linestyle='--', c='white')
    check = 0.5 * (ax[0].get_ylim()[1] - ax[0].get_ylim()[0]) + ax[0].get_ylim()[0]
    ax[0].text(result_epoch + 0.1, check, 'model', rotation=90, color='white')

    # Plot performance curves if they exist
    for p in range(len(perf_set)):
        for curve in perf_curves:
            if perf_set[p] in curve:
                plot_curves(ax[p+1], {curve: curves[curve]}, epochs, result_epoch, perf_set[p])
        ax[p+1].axvline(x=result_epoch, linestyle='--', c='white')
        check = 0.5 * (ax[p+1].get_ylim()[1] - ax[p+1].get_ylim()[0]) + ax[p+1].get_ylim()[0]
        ax[p+1].text(result_epoch + 0.1, check, 'model', rotation=90, color='white')


    # Save the figure
    save_path = os.path.join(model_viz_dir(exp_output_dir, model_name), 'learning_curves.png')
    plt.savefig(save_path, dpi=generic_dpi)
    plt.close()

def get_learning_curves(exp_output_dir: str, model_name: str) -> tuple:
    """
    Extract and compile learning curves from experiment logs for a specified model.

    Parameters
    ----------
    exp_output_dir : str
        Directory where experiment output is stored.
    model_name : str
        Name of the model to retrieve learning curves for.

    Returns
    -------
    curves : dict of str -> list of float or None
        A dictionary where each key represents a metric, and the corresponding value is a list
        of floats representing the metric values for each epoch.
    num_epochs : int
        Number of epochs trained for.
    """

    # get raw curve data from lightning logs
    curve_data = load_from_path(learning_data_path(exp_output_dir, model_name))

    # compile curve data into metric-wise dictionary
    curves = defaultdict(dict)
    for row in curve_data:
        for c in row:
            if row['epoch'] not in curves[c] or curves[c][row['epoch']] == '':
                curves[c][row['epoch']] = row[c]
    curves = dict(curves)

    # determine number of epochs trained for, and drop irrelevant metrics
    num_epochs = len(curves['epoch'])
    curves.pop('step')
    curves.pop('epoch')
    curve_names = list(curves.keys())
    for c in curve_names:
        if 'result' in c:
            curves.pop(c)

    # For each metric, make a list of float values in order of epochs
    for c in curves:
        new_curve = []
        for val in curves[c]:
            try:
                new_curve.append(float(curves[c][val]))
            except:
                new_curve.append(None)
        curves[c] = new_curve

    return curves, num_epochs

# ----------------------------------------------------------------------------------------------------------------------
# Visualize predictions
# ----------------------------------------------------------------------------------------------------------------------

def plot_set_predictions(exp_output_dir: str, model_name: str, data_tag: str) -> None:
    """
    Generates and saves prediction plots for a specific dataset and model output.

    This function fetches predictions and target values for each instance in a dataset,
    applies a visualization based on the model's task type (either regression or classification),
    and saves these plots to the designated visualization directory. For classification tasks,
    only a single output time step is currently supported.

    Parameters
    ----------
    exp_output_dir : str
        Path to the main experiment output directory where model results are stored.
    model_name : str
        Identifier for the specific model from which predictions are generated.
    data_tag : str
        A tag specifying the data subset (e.g., 'train', 'val', 'test') for which predictions
        are visualized.

    Notes
    -----
    - The function expects model arguments, including task type and output configurations,
      to be stored in a YAML file within the experiment output directory.
    - If the task type is 'classification' and more than one output time step is specified,
      an error is raised, as multiple output steps are unsupported in this mode.
    - Plots are saved in a dedicated directory within the experiment output structure.

    Raises
    ------
    SystemExit
        If the specified task type is unrecognized or if multiple output time steps
        are requested for a classification task.
    """

    # get model_args dictionary
    model_args = yaml_to_dict(model_args_path(exp_output_dir, model_name))

    # only single output timestep currently supported for classification
    out_range = model_args['data']['out_chunk'][1] - model_args['data']['out_chunk'][0]
    if out_range > 0 and model_args['data']['task'] == 'classification':
        logger.error('Cannot predict multiple output time steps if classification.')
        exit(101)

    # look up prediction directory for current experiment
    predictions_dir = model_predictions_dir(exp_output_dir, model_name)

    # find prediction and target values
    _, predictions, targets, timestamps  = get_predictions(predictions_dir, data_tag,
                                                           model_args['data_dynamics'][data_tag + '_size'])
    instances = list(set([timestamps[i][0] for i in timestamps]))

    # fill gaps where necessary
    predictions, targets = compile_predictions_for_plot(predictions, targets, timestamps, instances)

    # call the relevant plot function for each instance separately
    save_dir = model_viz_dir(exp_output_dir, model_name)
    for i in instances:
        if model_args['data']['task'] == 'regression':
            plot_regression_set_prediction(i, predictions, targets, data_tag,
                                      model_args['data']['out_components'], save_dir)
        elif model_args['data']['task'] == 'classification':
            plot_classification_set_prediction(i, predictions, targets, data_tag,
                                          save_dir, model_args['data_dynamics']['class_set'])
        else:
            logger.error('Unknown task type %s.', model_args['data']['task'])
            exit(101)


def compile_predictions_for_plot(predictions: dict, targets: dict, timestamps: dict, instances: list) -> tuple:
    """
    Organize predictions, targets, and timestamps for plotting, ensuring a consistent time axis with no gaps.

    This function takes in predictions, targets, and timestamps for multiple instances and returns
    structured dictionaries for plotting. It verifies that all time steps are consistent; if gaps
    are found, they are filled with NaN values to maintain continuity in the plot.

    Parameters
    ----------
    predictions : dict
        Dictionary mapping indices to predicted values for each instance.
    targets : dict
        Dictionary mapping indices to actual target values for each instance.
    timestamps : dict
        Dictionary mapping indices to corresponding timestamps for each instance.
    instances : list
        List of unique identifiers for instances to process.

    Returns
    -------
    tuple
        - ret_predictions : dict
            Dictionary where keys are instances, and values are lists containing arrays of sorted
            timestamps and corresponding predictions (with NaNs for any gaps).
        - ret_targets : dict
            Dictionary where keys are instances, and values are lists containing arrays of sorted
            timestamps and corresponding target values (with NaNs for any gaps).

    Raises
    ------
    SystemExit
        If unexpected time deltas cannot be resolved, logging an error and exiting with code 101.

    Notes
    -----
    - Ensures that any gaps in time steps are filled with NaNs to provide a continuous time series for plotting.
    - Identifies and sorts the relevant prediction points for each instance by timestamp.
    - Converts timezoned timestamps to UTC and removes timezones if detected, for uniform plotting.
    """
    ret_predictions = {}
    ret_targets = {}
    for i in instances:

        # find relevant prediction points and sort
        i_inx = [p for p in timestamps if timestamps[p][0] == i]
        t = [timestamps[p][2] for p in i_inx]
        y = [targets[p] for p in i_inx]
        y_hat = [predictions[p] for p in i_inx]
        order  = sorted(range(len(t)), key=t.__getitem__)
        t = np.array([t[x] for x in order])
        y = np.array([y[x] for x in order])
        y_hat = np.array([y_hat[x] for x in order])

        # check timezones
        if t.dtype == pd.Timestamp or t.dtype == datetime:
            timezones = np.array([ts.tz for ts in t])
            has_some_timezones = (timezones != None).any()
            has_some_nontimezones = (timezones == None).any()
            if has_some_timezones and has_some_nontimezones:
                logger.error('Instance time indices contains both timezoned and non-timezoned timestamps. This should not be possible at this point.')
                exit(101)
            if has_some_timezones:
                logger.warning('Instance has timezones, all timezones converted to UTC and dropped for plotting.')
                t = np.array([ts.tz_convert('UTC') for ts in t])
                t = np.array([ts.tz_localize(None) for ts in t])

        # check time deltas
        deltas = np.diff(t)
        delta = deltas.min()
        # if gaps exists, fill with nans
        if (deltas != delta).any():
            # prep a nan
            a_nice_nan = np.empty_like(y[0])
            a_nice_nan[:] = np.nan
            # find gaps
            gaps = []
            here = np.argwhere(deltas != delta)
            for h in range(len(here)):
                to_insert_nan = here[h].item()
                gap = np.arange(start=t[to_insert_nan] + delta,
                                stop=t[to_insert_nan + 1],
                                step=delta)
                gaps.append(gap)
            # fill gaps
            for g in range(len(gaps) - 1, 0, -1):
                to_insert_nan = here[g].item() + 1
                t = np.insert(t, to_insert_nan, gaps[g])
                nan_stack = np.stack([a_nice_nan for x in range(len(gaps[g]))], axis=0)
                y = np.insert(y, to_insert_nan, nan_stack, axis=0)
                y_hat = np.insert(y_hat, to_insert_nan, nan_stack, axis=0)
            if delta != np.diff(t).min():
                logger.error('Something went horribly wrong with growing gaps in prediction visuals.')
                exit(101)

        ret_predictions[i] = [t, y_hat]
        ret_targets[i] = [t, y]

    return ret_predictions, ret_targets


def plot_regression_set_prediction(i: any, predictions: dict, targets: dict, data_tag: str,
                              out_components: list, save_dir: str) -> None:
    """
    Plot regression predictions against target values for a specified instance across output components.

    This function generates and saves a plot for each output component, comparing the model's
    predictions with the target values over time. It includes Mean Absolute Error (MAE) for each
    time step, helping assess prediction accuracy visually.

    Parameters
    ----------
    i : any
        Unique identifier for the instance being plotted.
    predictions : dict
        Dictionary containing predictions for each instance, with keys representing indices.
    targets : dict
        Dictionary containing target values for each instance, with keys representing indices.
    data_tag : str
        Label for the dataset split being plotted (e.g., 'train', 'valid', 'eval').
    out_components : list
        List of output component names, used to label each component's plot.
    save_dir : str
        Directory where plots will be saved.

    Returns
    -------
    None
        Saves the generated plots directly to the specified directory.

    Notes
    -----
    - The function plots each output component of predictions and targets over time, including a
      calculated Mean Absolute Error (MAE) for each time step.
    - The plots are saved as PNG files with unique filenames incorporating the data tag and instance ID.
    """

    x = predictions[i][0]
    y_hat = predictions[i][1]
    y = targets[i][1]
    y_time = y_hat.shape[1]
    y_components = y_hat.shape[2]

    for c in range(y_components):
        fig, ax = plt.subplots(1, 1, figsize=large_figsize)
        for t in range(y_time):
            map = ax.plot(x, y[:, t, c],
                          label='Target ' + out_components[c] + ' step ' + str(t), color=target_color)
            mae = np.nanmean(np.abs(y[:, t, c] - y_hat[:, t, c]))
            ax.plot(x, y_hat[:, t, c],
                    label='Predicted ' + out_components[c] + ' step ' + str(t) + ' (mae=' + str(mae) + ')', color=predicted_color)
        ax.set_title(data_tag + ' instance: ' + str(i), fontsize=20)
        ax.set_xlabel('t', fontsize=20)
        ax.set_ylabel(str(out_components[c]) + '(t)', fontsize=20)
        ax.grid(color=grid_color, alpha=0.5)
        plt.legend(fontsize=10)
        plt.tight_layout()
        save_path = os.path.join(save_dir, data_tag + '-prediction-' + str(i) + '-' + str(out_components[c]) + '.png')
        plt.savefig(save_path, dpi=generic_dpi)
        plt.close()

def plot_classification_set_prediction(i: any, predictions: dict, targets: dict, data_tag: str, save_dir: str,
                                       class_set: dict) -> None:
    """
    Generate and save a confusion matrix plot for predictions versus targets in a classification task.

    This function creates a confusion matrix for a single instance's predictions and targets, calculates the
    instance's classification accuracy, and displays class labels from the provided class set. The plot is then
    saved to the specified directory.

    Parameters
    ----------
    i : any
        Identifier for the instance being plotted.
    predictions : dict
        Dictionary containing prediction arrays for each instance. Each array is expected to have shape [timesteps, classes].
    targets : dict
        Dictionary containing target arrays for each instance. Each array is expected to have shape [timesteps, classes].
    data_tag : str
        Identifier tag for the dataset or task, used in the plot title and file name.
    save_dir : str
        Directory path where the generated plot will be saved.
    class_set : dict
        Dictionary mapping class labels to class names or descriptions.

    Returns
    -------
    None
        Saves the plot to the specified directory as a PNG file. No output is returned.

    Notes
    -----
    - Computes the instance's accuracy, excluding any NaN values in the targets.
    - Handles NaNs in predictions and targets using `special_nanargmax`, which computes the argmax while ignoring NaNs.
    - Displays a confusion matrix with class labels for visual comparison of predicted and target classes.

    Raises
    ------
    FileNotFoundError
        If the save directory does not exist or cannot be accessed.
    """
    def special_nanargmax(arr, axis):
        new_arr = np.zeros(arr.shape[0]) + np.nan
        d0 = np.isnan(arr).all(axis=axis)
        new_arr[~d0] = np.nanargmax(arr[~d0], axis=1)
        return new_arr

    fig, ax = plt.subplots(1, 1, figsize=generic_figsize)
    y_hat = special_nanargmax(predictions[i][1], axis=1)
    y = special_nanargmax(targets[i][1], axis=1)
    class_labels = [c for c in class_set.keys()]
    num_classes = len(class_labels)

    correct = y == y_hat
    correct = 1. * correct
    nan_mask = np.isnan(y)
    correct[nan_mask] = np.nan
    accuracy = np.count_nonzero(correct[~nan_mask]) / float(len(correct[~nan_mask]))
    title = data_tag + ' instance: ' + str(i) + '; accuracy = ' + str(accuracy)

    conf_mat = np.zeros(shape=(num_classes, num_classes))
    for c_predicted in range(num_classes):
        for c_target in range(num_classes):
            t_hits = y == c_target
            p_hits = y_hat == c_predicted
            val = np.count_nonzero(np.logical_and(t_hits, p_hits))
            conf_mat[c_target, c_predicted] = val

    ConfusionMatrixDisplay(confusion_matrix=conf_mat,
                           display_labels=class_labels).plot(ax=ax, cmap=generic_cmap, im_kw={'aspect': 'auto'})

    plt.title(title)
    plt.tight_layout()
    save_path = os.path.join(save_dir, data_tag + '-prediction-' + str(i) + '.png')
    plt.savefig(save_path, dpi=generic_dpi)
    plt.close()

# ----------------------------------------------------------------------------------------------------------------------
# Visualize interpretations
# TODO: Lots of opportunity for refactoring here
# ----------------------------------------------------------------------------------------------------------------------

def plot_feature_attribution(exp_output_dir: str, model_name: str, interpretation_file_name: str) -> None:
    """
    Generates and saves visualizations for feature attribution analysis based on model type.

    Parameters:
    -----------
    exp_output_dir : str
        The directory path to the experiment output containing model configurations and results.
    model_name : str
        The name of the model for which feature attributions are visualized.
    interpretation_file_name : str
        The file name of the interpretation data to be loaded and visualized.

    Returns:
    --------
    None
        The function generates and saves various feature attribution visualizations (animations
        and heatmaps) based on the model's task (regression or classification).

    Notes:
    ------
    - This function reads model arguments and feature attribution data, then creates visualizations
      according to the task type (either 'regression' or 'classification').
    - Visualizations include mean feature attributions, running animations, and classic
      animations for either regression or classification tasks.
    - Visualizations are saved in the specified model's visualization directory.

    """
    interpretation_dir = model_interpretations_dir(exp_output_dir, model_name)
    model_args = yaml_to_dict(model_args_path(exp_output_dir, model_name))
    file_path = os.path.join(interpretation_dir, interpretation_file_name)
    feat_att_dict = load_from_path(file_path)
    save_dir = model_viz_dir(exp_output_dir, model_name)

    if model_args['data']['task'] == 'regression':
        running_animation_regression(feat_att_dict, save_dir, model_args, interpretation_file_name)
        mean_feat_att_regression(feat_att_dict, save_dir, model_args, interpretation_file_name)
        running_animation_regression(feat_att_dict, save_dir, model_args, interpretation_file_name, classic=True)

    elif model_args['data']['task'] == 'classification':
        running_animation_classification(feat_att_dict, save_dir, model_args, interpretation_file_name)
        mean_feat_att_classification(feat_att_dict, save_dir, model_args, interpretation_file_name)
        running_animation_classification(feat_att_dict, save_dir, model_args, interpretation_file_name, classic=True)

def running_animation_classification(feat_att_dict: dict, save_dir: str, model_args: dict,
                                     interpretation_file_name: str, classic: bool = False) -> None:
    """
    Generates animated visualizations for classification model interpretations and saves them as .gif files.

    This function creates animations to show the time series evolution of predictions, targets, and feature
    attributions for each class label over specified instances. The animations are saved in .gif format for
    visualizing model behavior across time or sequential steps.

    Parameters
    ----------
    feat_att_dict : dict
        A dictionary containing model interpretation details, including:
            - 'results': Dictionary of feature attributions, indexed by each class label.
            - 'timestamps': List of timestamps relevant to each feature attribution entry.
            - 'predictions': Array of model predictions for each timestamp.
            - 'targets': Array of true target values for each timestamp.
            - 'input_features': Array of input features for each timestamp.
    save_dir : str
        The directory path where the animated .gif files will be saved.
    model_args : dict
        Dictionary of model and dataset metadata, specifically:
            - 'data': A dictionary with keys 'in_components', 'in_chunk', 'meta_path', and 'package_path' to retrieve time delta.
            - 'data_dynamics': A dictionary containing 'class_set' which maps each class to its corresponding index.
    interpretation_file_name : str
        The filename of the interpretation data, used as the base name for saving .gif files.
    classic : bool, default = False
        Whether to use the old animation style.

    Returns
    -------
    None
        The function saves the generated animations as .gif files in the specified directory.

    Notes
    -----
    - Each animation illustrates the evolution of predictions and targets for a given class label and highlights
      input feature attributions that exceed a 95th percentile threshold.
    - The function requires `compile_running_plot_animation` to compile and save animations.
    """

    def scale_alpha(min_val, max_val, val, epsilon=1e-6):
        def lin_scale(f, target_min, target_max, native_min, native_max):
            return (target_max - target_min) * (f - native_min) / (native_max - native_min + epsilon) + target_min
        alpha = lin_scale(val, 0, 1, min_val, max_val)
        return alpha

    interpretation_name = os.path.splitext(interpretation_file_name)[0]

    # get meta info about dataset that was interpreted on
    in_comps = model_args['data']['in_components']
    in_chunk = model_args['data']['in_chunk']
    in_scan = np.arange(in_chunk[0], in_chunk[-1] + 1)
    t_delta = BaseDataset(model_args['data']['meta_path'], model_args['data']['package_path']).time_delta
    t_delta = np.timedelta64(t_delta)

    class_set = model_args['data_dynamics']['class_set']

    # find instances relevant to current interpretation
    i = np.array([feat_att_dict['timestamps'][t][0] for t in range(len(feat_att_dict['timestamps']))])
    instances = list(set(i))

    # for every output class
    for c in class_set:
        logit = class_set[c]
        # for every relevant instance
        for instance in instances:
            relevant_to_i = np.argwhere(i == instance).squeeze()
            s = np.array([feat_att_dict['timestamps'][pp][1] for pp in relevant_to_i])
            slices = list(set(s))
            # for every relevant (to current instance) slice
            for slice in slices:
                relevant_to_s = relevant_to_i[np.argwhere(s == slice).squeeze()]
                y = np.array(feat_att_dict['targets'])[relevant_to_s]
                y_hat = np.array(feat_att_dict['predictions'])[relevant_to_s]
                t = np.array(feat_att_dict['timestamps'])[relevant_to_s][:, 2]
                t = [c for c in t]
                plot_data = []
                ts_tick = 0
                for ts in relevant_to_s:
                    new_plot_data = {}
                    new_plot_data['prediction_curve'] = (t, y_hat[:, logit])
                    new_plot_data['target_curve'] = (t, y[:, logit])
                    new_plot_data['pp_tick'] = t[ts_tick]
                    new_plot_data['chunk_0_tick'] = t[ts_tick] + t_delta * in_scan[0]
                    new_plot_data['chunk_1_tick'] = t[ts_tick] + t_delta * in_scan[-1]
                    min_f = feat_att_dict['results'][logit]['attributions'][ts, :, :].abs().min().item()
                    max_f = feat_att_dict['results'][logit]['attributions'][ts, :, :].abs().max().item()
                    upper_threshold = np.percentile(feat_att_dict['results'][logit]['attributions'][ts, :, :].abs().detach().cpu().numpy(), 95)
                    ic_curves = defaultdict(list)
                    new_plot_data['ic_curve_attributions'] = defaultdict(list)
                    for ic in range(len(in_comps)):
                        for in_delay in range(1 + in_chunk[1] - in_chunk[0]):
                            feature_attribution_val = feat_att_dict['results'][logit]['attributions'][ts, in_delay, ic].abs().cpu().item()
                            alpha = scale_alpha(min_f, max_f, feature_attribution_val)
                            if feature_attribution_val > upper_threshold:
                                alpha=2.
                            if not classic:
                                new_plot_data['ic_curve_attributions'][ic].append(alpha)
                            else:
                                new_plot_data['in_scan'] = in_scan
                                new_plot_data['ic_curve_attributions'][ic].append(feature_attribution_val)
                            ic_curves[ic].append([t[ts_tick] + t_delta * in_scan[in_delay], feat_att_dict['input_features'][ts, in_delay, ic].item()])
                    new_plot_data['ic_curves'] = ic_curves
                    plot_data.append(new_plot_data)
                    ts_tick += 1
                if not classic:
                    file_name = interpretation_name + '_running-i=' + str(instance) + '-s=' + str(
                        slice) + '-class=' + str(c) + '.gif'
                    plot_save_path = os.path.join(save_dir, file_name)
                    compile_running_plot_animation(plot_data, plot_save_path, in_comps)
                else:
                    file_name = interpretation_name + '_classic-i=' + str(instance) + '-s=' + str(
                        slice) + '-class=' + str(c) + '.gif'
                    plot_save_path = os.path.join(save_dir, file_name)
                    compile_classic_plot_animation(plot_data, plot_save_path, in_comps)

def running_animation_regression(feat_att_dict: dict, save_dir: str, model_args: dict,
                                 interpretation_file_name: str, classic: bool = False) -> None:
    """
    Generates animated visualizations for model interpretations and saves them as .gif files.

    This function takes feature attributions, prediction and target curves, timestamps, and model
    meta information to generate and save animated plots illustrating running predictions over time.
    Each output component's temporal behavior is animated based on the provided time series data.

    Parameters
    ----------
    feat_att_dict : dict
        Dictionary containing feature attributions, predictions, targets, and timestamps for
        model interpretations. The expected keys include:
            - 'results': Dictionary with attribution data, indexed by tuples `(output_delay, component)`.
            - 'timestamps': List of timestamps for each feature attribution entry.
            - 'predictions': Array of model predictions.
            - 'targets': Array of true target values.
            - 'input_features': Array of input feature values.
    save_dir : str
        Directory path where the animated .gif files will be saved.
    model_args : dict
        Dictionary containing metadata about the dataset and model parameters, specifically:
            - 'data': A dictionary with keys 'in_components', 'out_components', 'in_chunk', 'out_chunk',
              and 'meta_path' for loading time delta and related settings.
    interpretation_file_name : str
        The filename of the interpretation data, used to generate the .gif filenames.
    classic : bool, default = False
        Whether to use the old animation style.

    Returns
    -------
    None
        The function saves the generated animations as .gif files in the specified directory.

    Notes
    -----
    - Each animation visualizes predictions, targets, and input component attributions for a specified
      output component and delay, highlighting absolute feature attributions that exceed the 95th percentile.
    - The visibility of the markers are based on their absolute feature attributions.
        There is a linear scaling from least important (invisible) to most important (completely visible).
    - Markers that marked with a star are those whose absolute feature attribution is above the 95th percentile of
        all feature attributions at that point in time.
    - The helper function `compile_running_plot_animation` is required to compile and save the animations.
    """

    def scale_alpha(min_val, max_val, val):
        def lin_scale(f, target_min, target_max, native_min, native_max):
            return (target_max - target_min) * (f - native_min) / (native_max - native_min) + target_min
        alpha = lin_scale(val, 0, 1, min_val, max_val)
        return alpha

    interpretation_name = os.path.splitext(interpretation_file_name)[0]

    # get meta info about dataset that was interpreted on
    in_comps = model_args['data']['in_components']
    out_comps = model_args['data']['out_components']
    in_chunk = model_args['data']['in_chunk']
    out_chunk = model_args['data']['out_chunk']
    in_scan = np.arange(in_chunk[0], in_chunk[-1] + 1)
    t_delta = BaseDataset(model_args['data']['meta_path'], model_args['data']['package_path']).time_delta
    t_delta = np.timedelta64(t_delta)

    # find instances relevant to current interpretation
    i = np.array([feat_att_dict['timestamps'][t][0] for t in range(len(feat_att_dict['timestamps']))])
    instances = list(set(i))

    # for every output component
    for oc in range(len(out_comps)):
        # for every output delay
        for out_delay in range(1 + out_chunk[1] - out_chunk[0]):
            logit = (out_delay, oc)
            # for every relevant instance
            for instance in instances:
                relevant_to_i = np.argwhere(i == instance).squeeze()
                if relevant_to_i.size == 1:
                    relevant_to_i = np.array([relevant_to_i])
                s = np.array([feat_att_dict['timestamps'][pp][1] for pp in relevant_to_i])
                slices = list(set(s))
                # for every relevant (to current instance) slice
                for slice in slices:
                    relevant_to_s = relevant_to_i[np.argwhere(s == slice).squeeze()]
                    if relevant_to_s.size == 1:
                        relevant_to_s = np.array([relevant_to_s])
                    y = np.array(feat_att_dict['targets'])[relevant_to_s]
                    y_hat = np.array(feat_att_dict['predictions'])[relevant_to_s]
                    t = np.array(feat_att_dict['timestamps'])[relevant_to_s][:, 2]
                    t = [c for c in t]
                    plot_data = []
                    ts_tick = 0
                    for ts in relevant_to_s:
                        new_plot_data = {}
                        new_plot_data['prediction_curve'] = (t, y_hat[:, out_delay, oc])
                        new_plot_data['target_curve'] = (t, y[:, out_delay, oc])
                        new_plot_data['pp_tick'] = t[ts_tick]
                        new_plot_data['chunk_0_tick'] = t[ts_tick] + t_delta * in_scan[0]
                        new_plot_data['chunk_1_tick'] = t[ts_tick] + t_delta * in_scan[-1]
                        min_f = feat_att_dict['results'][logit]['attributions'][ts, :, :].abs().min().item()
                        max_f = feat_att_dict['results'][logit]['attributions'][ts, :, :].abs().max().item()
                        upper_threshold = np.percentile(feat_att_dict['results'][logit]['attributions'][ts, :, :].abs().detach().cpu().numpy(), 95)
                        ic_curves = defaultdict(list)
                        new_plot_data['ic_curve_attributions'] = defaultdict(list)
                        for ic in range(len(in_comps)):
                            for in_delay in range(1 + in_chunk[1] - in_chunk[0]):
                                feature_attribution_val = feat_att_dict['results'][logit]['attributions'][ts, in_delay, ic].abs().cpu().item()
                                alpha = scale_alpha(min_f, max_f, feature_attribution_val)
                                if feature_attribution_val > upper_threshold:
                                    alpha=2.
                                if not classic:
                                    new_plot_data['ic_curve_attributions'][ic].append(alpha)
                                else:
                                    new_plot_data['in_scan'] = in_scan
                                    new_plot_data['ic_curve_attributions'][ic].append(feature_attribution_val)
                                ic_curves[ic].append([t[ts_tick] + t_delta * in_scan[in_delay], feat_att_dict['input_features'][ts, in_delay, ic].item()])
                        new_plot_data['ic_curves'] = ic_curves
                        plot_data.append(new_plot_data)
                        ts_tick += 1
                    if not classic:
                        file_name = interpretation_name + '_running-i=' + str(instance) + '-s=' + str(
                            slice) + '-logit=' + str(logit) + '.gif'
                        plot_save_path = os.path.join(save_dir, file_name)
                        compile_running_plot_animation(plot_data, plot_save_path, in_comps)
                    else:
                        file_name = interpretation_name + '_classic-i=' + str(instance) + '-s=' + str(
                            slice) + '-logit=' + str(logit) + '.gif'
                        plot_save_path = os.path.join(save_dir, file_name)
                        compile_classic_plot_animation(plot_data, plot_save_path, in_comps)

def compile_running_plot_animation(plot_data: list, save_path: str, in_comps: list) -> None:
    """
    Creates and saves an animated plot showing predictions, targets, and input feature contributions over time.

    This function generates an animated plot that visualizes model predictions and target values, with
    additional markers highlighting input feature attributions over a time series. The resulting animation
    is saved as a .gif file, providing insight into model behavior across sequential time steps.

    Parameters
    ----------
    plot_data : list of dict
        A list of dictionaries containing plotting data for each time step. Each dictionary includes:
            - 'prediction_curve': Tuple of (time, prediction values).
            - 'target_curve': Tuple of (time, target values).
            - 'pp_tick': Scalar, timestamp for the prediction point.
            - 'chunk_0_tick' and 'chunk_1_tick': Scalars, timestamps defining input chunk range.
            - 'ic_curves': Dictionary of input component time series (one list per component).
            - 'ic_curve_attributions': Dictionary of transparency levels for each component at each time step.
    save_path : str
        Path where the animated .gif file will be saved.
    in_comps : list of str
        List of names or identifiers for each input component, used for labeling in the legend.

    Returns
    -------
    None
        Saves an animation to the specified `save_path`.

    Notes
    -----
    - Each input feature component curve is represented with varying opacity to indicate feature importance.
    - Only updates specific plot elements on each animation frame for efficient rendering.
    - The function requires `pillow` for .gif saving.
    """

    def get_full_color_cycle(options):
        """"Generates a color cycle list for input components based on the given color options."""
        colors = []
        c_tick = 0
        for c in range(len(options)):
            colors.append(color_cycle[c_tick])
            c_tick += 1
            if c_tick == len(color_cycle) - 1:
                c_tick = 0
        return colors

    def construct_custom_legend(in_comps, colors):
        """Creates a custom legend with prediction, target, and input component labels."""
        custom_lines = [Line2D([0], [0], color=get_color('predicted'), lw=4)]
        custom_names = ['prediction']
        custom_lines.append(Line2D([0], [0], color=get_color('target'), lw=4))
        custom_names.append('target')
        for c in range(len(in_comps)):
            custom_lines.append(Line2D([0], [0], color=colors[c], lw=4))
            custom_names.append(in_comps[c])
        return custom_lines, custom_names

    def update(step):
        """Updates plot elements to animate each frame based on current time step data."""
        p = plot_data[step]
        prediction_curve.set_xdata(p['prediction_curve'][0])
        prediction_curve.set_ydata(p['prediction_curve'][1])
        target_curve.set_xdata(p['target_curve'][0])
        target_curve.set_ydata(p['target_curve'][1])
        pp_tick.set_xdata(p['pp_tick'])
        chunk_0_tick.set_xdata(p['chunk_0_tick'])
        chunk_1_tick.set_xdata(p['chunk_1_tick'])
        ticker = 0
        for ic in range(len(ic_curves)):
            t = [a[0] for a in p['ic_curves'][ic]]
            f = [b[1] for b in p['ic_curves'][ic]]
            ic_curves[ic].set_xdata(t)
            ic_curves[ic].set_ydata(f)
            for ts in range(len(t)):
                ic_dots[ticker].set_xdata(t[ts])
                ic_dots[ticker].set_ydata(f[ts])
                alpha = p['ic_curve_attributions'][ic][ts]
                if alpha <= 1.0:
                    ic_dots[ticker].set(alpha=alpha, marker='o', markersize=10.)
                else:
                    ic_dots[ticker].set(alpha=1.0, marker='*', markersize=20.)
                ticker += 1

        return_val = [prediction_curve, target_curve, pp_tick, chunk_0_tick, chunk_1_tick]
        for ic in range(len(ic_curves)):
            return_val.append(ic_curves[ic])
        for d in range(len(ic_dots)):
            return_val.append(ic_dots[d])

        return return_val

    colors = get_full_color_cycle(in_comps)
    custom_lines, custom_names = construct_custom_legend(in_comps, colors)

    # create initial plot
    fig, ax = plt.subplots(1, 1, figsize=generic_figsize, dpi=generic_dpi)
    p = plot_data[0]

    prediction_curve = ax.plot(p['prediction_curve'][0], p['prediction_curve'][1], c=get_color('predicted'),)[0]
    target_curve = ax.plot(p['target_curve'][0], p['target_curve'][1], c=get_color('target'))[0]
    pp_tick = ax.axvline(x=p['pp_tick'], color="grey", linestyle='-', alpha=0.5)
    chunk_0_tick = ax.axvline(x=p['chunk_0_tick'], color="grey", linestyle='--', alpha=0.5)
    chunk_1_tick = ax.axvline(x=p['chunk_1_tick'], color="grey", linestyle='--', alpha=0.5)
    ax.legend(custom_lines, custom_names)
    ic_curves = []
    ic_dots = []
    for ic in p['ic_curves']:
        t = [a[0] for a in p['ic_curves'][ic]]
        f = [b[1] for b in p['ic_curves'][ic]]
        ic_curves.append(ax.plot(t, f, c=colors[ic], alpha=0.3)[0])
        for ts in range(len(t)):
            alpha = p['ic_curve_attributions'][ic][ts]
            if alpha <= 1.0:
                ic_dots.append(ax.plot(t[ts], f[ts], c=colors[ic])[0])
                ic_dots[-1].set(alpha=alpha, marker='o', markersize=10.)
            else:
                ic_dots.append(ax.plot(t[ts], f[ts], c=colors[ic])[0])
                ic_dots[-1].set(alpha=1.0, marker='*', markersize=20.)

    ax.set_xlabel('prediction points')
    ax.set_ylabel('feature value')
    animation_fig = animation.FuncAnimation(fig, update,
                                            frames=len(plot_data),
                                            interval=200,
                                            blit=True,
                                            repeat_delay=10,
                                            repeat=True, )

    animation_fig.save(save_path, writer="pillow")
    plt.close()

def compile_classic_plot_animation(plot_data: list, save_path: str, in_comps: list) -> None:
    """
    Creates and saves an animated plot of feature attributions and input component curves over time.

    Parameters:
    -----------
    plot_data : list
        A list of dictionaries containing the data for each frame of the animation, including
        prediction curves, target curves, input component curves, and feature attribution values.
    save_path : str
        The file path where the animation will be saved, usually with a `.gif` or `.mp4` extension.
    in_comps : list
        List of input component names used for labelling and color-coding input component curves.

    Returns:
    --------
    None
        The function saves an animation file at `save_path`.

    Notes:
    ------
    - This function generates an animation where each frame is updated with new prediction, target,
      and input component data.
    - The function customizes the plot legend, color-coding each input component curve distinctly.
    - Uses `update` as an animation function to update each frame, iterating through data in `plot_data`.
    - A color bar indicating feature attribution intensity is displayed alongside the animation.

    """
    def get_full_color_cycle(options):
        """"Generates a color cycle list for input components based on the given color options."""
        colors = []
        c_tick = 0
        for c in range(len(options)):
            colors.append(color_cycle[c_tick])
            c_tick += 1
            if c_tick == len(color_cycle) - 1:
                c_tick = 0
        return colors

    def construct_custom_legend(in_comps, colors):
        """Creates a custom legend with prediction, target, and input component labels."""
        custom_lines = [Line2D([0], [0], color=get_color('predicted'), lw=4)]
        custom_names = ['prediction']
        custom_lines.append(Line2D([0], [0], color=get_color('target'), lw=4))
        custom_names.append('target')
        for c in range(len(in_comps)):
            custom_lines.append(Line2D([0], [0], color=colors[c], lw=4))
            custom_names.append(in_comps[c])
        return custom_lines, custom_names

    def update(step):
        """Updates plot elements to animate each frame based on current time step data."""
        p = plot_data[step]
        prediction_curve.set_xdata(p['prediction_curve'][0])
        prediction_curve.set_ydata(p['prediction_curve'][1])
        target_curve.set_xdata(p['target_curve'][0])
        target_curve.set_ydata(p['target_curve'][1])
        pp_tick.set_xdata(p['pp_tick'])
        chunk_0_tick.set_xdata(p['chunk_0_tick'])
        chunk_1_tick.set_xdata(p['chunk_1_tick'])
        h_map = []
        for ic in range(len(ic_curves)):
            t = [a[0] for a in p['ic_curves'][ic]]
            f = [b[1] for b in p['ic_curves'][ic]]
            ic_curves[ic].set_xdata(t)
            ic_curves[ic].set_ydata(f)
            h_map.append(p['ic_curve_attributions'][ic])
        h_map = np.array(h_map)
        feat_att_map.set_data(h_map)

        return_val = [prediction_curve, target_curve, pp_tick, chunk_0_tick, chunk_1_tick]
        for ic in range(len(ic_curves)):
            return_val.append(ic_curves[ic])
        return_val.append(feat_att_map)

        return return_val

    colors = get_full_color_cycle(in_comps)
    custom_lines, custom_names = construct_custom_legend(in_comps, colors)

    # create initial plot
    fig, axes = plt.subplots(2, 1, figsize=generic_figsize, dpi=generic_dpi)
    p = plot_data[0]
    ax = axes[0]
    fax = axes[1]
    prediction_curve = ax.plot(p['prediction_curve'][0], p['prediction_curve'][1], c=get_color('predicted'),)[0]
    target_curve = ax.plot(p['target_curve'][0], p['target_curve'][1], c=get_color('target'))[0]
    pp_tick = ax.axvline(x=p['pp_tick'], color="grey", linestyle='-', alpha=0.5)
    chunk_0_tick = ax.axvline(x=p['chunk_0_tick'], color="grey", linestyle='--', alpha=0.5)
    chunk_1_tick = ax.axvline(x=p['chunk_1_tick'], color="grey", linestyle='--', alpha=0.5)
    ax.legend(custom_lines, custom_names)
    ic_curves = []
    h_map = []
    for ic in p['ic_curves']:
        t = [a[0] for a in p['ic_curves'][ic]]
        f = [b[1] for b in p['ic_curves'][ic]]
        ic_curves.append(ax.plot(t, f, c=colors[ic], alpha=0.9)[0])
        h_map.append(p['ic_curve_attributions'][ic])
    h_map = np.array(h_map)
    feat_att_map = fax.imshow(h_map, cmap=generic_cmap, aspect='auto')
    fax.set_xlabel('input delay')
    id_ticks, id_ticklabels = get_tick_params(p['in_scan'])
    fax.set_xticks(id_ticks)
    fax.set_xticklabels(id_ticklabels)
    fax.set_yticks([x for x in range(len(in_comps))])
    fax.set_yticklabels(in_comps)
    divider = make_axes_locatable(fax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(feat_att_map, cax=cax, orientation='vertical')
    cbar.set_label('feature attribution')
    ax.set_xlabel('prediction points')
    ax.set_ylabel('feature value')
    animation_fig = animation.FuncAnimation(fig, update,
                                            frames=len(plot_data),
                                            interval=200,
                                            blit=True,
                                            repeat_delay=10,
                                            repeat=True, )

    animation_fig.save(save_path, writer="pillow")
    plt.close()

def mean_feat_att_regression(feat_att_dict: dict, save_dir: str, model_args: dict, interpretation_file_name: str) -> None:
    """
    Processes feature attributions for a regression model and generates mean attribution heatmaps for each
    output component and delay.

    Parameters:
    -----------
    feat_att_dict : dict
        Dictionary containing feature attribution data, including timestamps and attributions for specific instances.
    save_dir : str
        Directory path where the generated mean attribution plot images will be saved.
    model_args : dict
        Dictionary of model parameters and metadata, containing keys for input and output components,
        chunk ranges, and data paths.
    interpretation_file_name : str
        Name of the interpretation file, used to label the generated images.

    Returns:
    --------
    None
        The function saves mean feature attribution plots as PNG files in `save_dir`.

    Notes:
    ------
    This function:
      - Extracts metadata from `model_args` to identify input components, chunks, and temporal deltas.
      - Loops through each output component and delay to compute relevant feature attributions.
      - Generates mean feature attribution heatmaps for each output component-delay-instance combination.
      - Uses the `plot_mean` function to create and save each plot.

    """
    interpretation_name = os.path.splitext(interpretation_file_name)[0]

    # get meta info about dataset that was interpreted on
    in_comps = model_args['data']['in_components']
    out_comps = model_args['data']['out_components']
    in_chunk = model_args['data']['in_chunk']
    out_chunk = model_args['data']['out_chunk']
    in_scan = np.arange(in_chunk[0], in_chunk[-1] + 1)

    # find instances relevant to current interpretation
    i = np.array([feat_att_dict['timestamps'][t][0] for t in range(len(feat_att_dict['timestamps']))])
    instances = list(set(i))

    # for every output component
    for oc in range(len(out_comps)):
        # for every output delay
        for out_delay in range(1 + out_chunk[1] - out_chunk[0]):
            logit = (out_delay, oc)
            # for every relevant instance
            for instance in instances:
                relevant_to_i = np.argwhere(i == instance).squeeze()
                if relevant_to_i.size == 1:
                    relevant_to_i = np.array([relevant_to_i])
                s = np.array([feat_att_dict['timestamps'][pp][1] for pp in relevant_to_i])
                slices = list(set(s))
                # for every relevant (to current instance) slice
                for slice in slices:
                    relevant_to_s = relevant_to_i[np.argwhere(s == slice).squeeze()]
                    if relevant_to_s.size == 1:
                        relevant_to_s = np.array([relevant_to_s])
                    t = np.array(feat_att_dict['timestamps'])[relevant_to_s][:, 2]
                    t = [c for c in t]
                    feature_attributions = feat_att_dict['results'][logit]['attributions'][relevant_to_s, :, :].abs().detach().cpu().numpy()
                    file_name = interpretation_name + '_simple_mean-i=' + str(instance) + '-s=' + str(
                        slice) + '-logit=' + str(logit) + '.png'
                    plot_save_path = os.path.join(save_dir, file_name)
                    plot_mean(t, feature_attributions, plot_save_path, in_comps, in_scan)

def mean_feat_att_classification(feat_att_dict: dict, save_dir: str, model_args: dict, interpretation_file_name: str) -> None:
    """
    Generates mean feature attribution heatmaps for each output class in a classification model.

    Parameters:
    -----------
    feat_att_dict : dict
        Dictionary containing feature attribution data, including timestamps, targets, and attributions for each instance.
    save_dir : str
        Directory path where the generated mean attribution plot images will be saved.
    model_args : dict
        Dictionary of model parameters and metadata, containing information on input and output components,
        chunk ranges, and data dynamics (such as class mappings).
    interpretation_file_name : str
        Name of the interpretation file, used to label the generated images.

    Returns:
    --------
    None
        The function saves mean feature attribution plots as PNG files in `save_dir`.

    Notes:
    ------
    - This function loops through each class and instance, generating mean feature attribution plots
      for each class-instance-slice combination.
    - Uses metadata from `model_args` to configure input components, chunks, and class mappings.
    - Calls the `plot_mean` function to create and save each heatmap plot based on the calculated mean
      feature attributions.

    """
    interpretation_name = os.path.splitext(interpretation_file_name)[0]

    # get meta info about dataset that was interpreted on
    in_comps = model_args['data']['in_components']
    out_comps = model_args['data']['out_components']
    in_chunk = model_args['data']['in_chunk']
    out_chunk = model_args['data']['out_chunk']
    in_scan = np.arange(in_chunk[0], in_chunk[-1] + 1)

    class_set = model_args['data_dynamics']['class_set']

    # find instances relevant to current interpretation
    i = np.array([feat_att_dict['timestamps'][t][0] for t in range(len(feat_att_dict['timestamps']))])
    instances = list(set(i))

    # for every output class
    for c in class_set:
        logit = class_set[c]
        # for every relevant instance
        for instance in instances:
            relevant_to_i = np.argwhere(i == instance).squeeze()
            s = np.array([feat_att_dict['timestamps'][pp][1] for pp in relevant_to_i])
            slices = list(set(s))
            # for every relevant (to current instance) slice
            for slice in slices:
                relevant_to_s = relevant_to_i[np.argwhere(s == slice).squeeze()]
                t = np.array(feat_att_dict['timestamps'])[relevant_to_s][:, 2]
                t = [c for c in t]
                feature_attributions = feat_att_dict['results'][logit]['attributions'][relevant_to_s, :, :].abs().detach().cpu().numpy()
                file_name = interpretation_name + '_simple_mean-i=' + str(instance) + '-s=' + str(
                    slice) + '-logit=' + str(logit) + '.png'
                plot_save_path = os.path.join(save_dir, file_name)
                plot_mean(t, feature_attributions, plot_save_path, in_comps, in_scan)

def plot_mean(t: list, feature_attributions: np.array, save_path: str, in_comps: list, in_scan: list) -> None:
    """
    Generates and saves a multi-panel heatmap plot to visualize the mean feature attributions
    across prediction points, input delays, and input components.

    Parameters:
    -----------
    t : array-like
        Array of time steps or prediction points.
    feature_attributions : ndarray
        3D array of feature attributions with dimensions (prediction points, input delays, input components).
    save_path : str
        Path where the generated plot image will be saved.
    in_comps : list
        List of input component names to label the y-axis of heatmaps.
    in_scan : array-like
        Array of input delay values corresponding to each input delay axis tick.

    Returns:
    --------
    None
        The function saves the generated plot as an image file at the specified `save_path`.

    Notes:
    ------
    This function performs the following steps:
      - Calculates mean values of `feature_attributions` across prediction points, input delays, and components.
      - Creates three heatmaps representing:
          1. Mean feature attribution across prediction points (top-left).
          2. Mean feature attribution across input delays (top-right).
          3. Mean feature attribution across input components (bottom-left).
      - Sets custom tick labels for each subplot.
      - Adds a shared color bar at the bottom of the plot to represent absolute attribution values.
      - Adjusts x and y labels to be positioned midway between subplots.

    """

    # collect relevant heatmap grids and feature ranges
    mean_across_pp = np.mean(feature_attributions, axis=0)
    mean_across_delays = np.mean(feature_attributions, axis=1)
    mean_across_comp = np.mean(feature_attributions, axis=2)
    max_mean = max(np.max(mean_across_pp), np.max(mean_across_delays), np.max(mean_across_comp))
    min_mean = min(np.min(mean_across_pp), np.min(mean_across_delays), np.min(mean_across_comp))

    # determine prediction point and input delay ticks
    t = np.array(t)
    pp_ticks, pp_ticklabels = get_tick_params(t)
    id_ticks, id_ticklabels = get_tick_params(in_scan)

    # initialize figure
    fig, ax = plt.subplots(2, 2, figsize=large_figsize, dpi=generic_dpi)

    # plot cross prediction point heatmap and set visual parameters
    map1 = ax[0, 0].imshow(mean_across_pp.transpose(),
                           aspect='auto', cmap=generic_cmap,
                           vmax=max_mean, vmin=min_mean)
    ax[0, 0].set_yticks([y for y in range(len(in_comps))])
    ax[0, 0].set_yticklabels(in_comps)
    ax[0, 0].set_xticks(id_ticks)
    ax[0, 0].set_xticklabels(id_ticklabels)
    ax[0, 0].yaxis.set_label_position('right')
    ax[0, 0].set_xlabel('input delay', fontsize='x-large')
    ax[0, 0].set_ylabel('input component', fontsize='x-large')
    ax[0, 0].tick_params(right=False, labelright=False,
                         top=True, labeltop=True,
                         bottom=False, labelbottom=False)

    # plot cross input delay heatmap and set visual parameters
    ax[0, 1].imshow(mean_across_delays.transpose(),
                           aspect='auto', cmap=generic_cmap,
                           vmax=max_mean, vmin=min_mean)
    ax[0, 1].set_yticks([y for y in range(len(in_comps))])
    ax[0, 1].set_yticklabels(in_comps)
    ax[0, 1].set_xticks(pp_ticks)
    ax[0, 1].set_xticklabels(pp_ticklabels, rotation=90)
    ax[0, 1].xaxis.set_label_position('top')
    ax[0, 1].set_xlabel('prediction point', fontsize='x-large')
    ax[0, 1].tick_params(right=True, labelright=True,
                         left=False, labelleft=False)

    # plot cross input component heatmap and set visual parameters
    ax[1, 0].imshow(mean_across_comp,
                           aspect='auto', cmap=generic_cmap,
                           vmax=max_mean, vmin=min_mean)
    ax[1, 0].set_yticks(pp_ticks)
    ax[1, 0].set_yticklabels(pp_ticklabels)
    ax[1, 0].set_xticks(id_ticks)
    ax[1, 0].set_xticklabels(id_ticklabels)
    ax[1, 0].set_ylabel('prediction point', fontsize='x-large')
    ax[1, 0].tick_params(right=True, labelright=True,
                         left=False, labelleft=False,
                         bottom=True, labelbottom=True)

    # delete fourth subplot and replace with global color bar
    fig.delaxes(ax[1, 1])
    divider = make_axes_locatable(ax[1, 1])
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    cbar = fig.colorbar(map1, cax=cax, orientation='horizontal')
    cbar.set_label('absolute feature attribution')

    # move shared x and y labels to midpoint between plots
    pos1 = ax[0, 0].get_position()
    pos2 = ax[0, 1].get_position()
    pos3 = ax[1, 0].get_position()
    midpoint_x = (pos1.x1 + pos2.x0) / 2
    midpoint_y = (pos1.y1 + pos2.y0) / 2
    ax[0, 0].yaxis.set_label_coords(midpoint_x, midpoint_y, transform=fig.transFigure)
    midpoint_x = (pos1.x1 + pos3.x0) / 2
    midpoint_y = (pos1.y1 + pos3.y0) / 2
    ax[0, 0].xaxis.set_label_coords(midpoint_x, midpoint_y, transform=fig.transFigure)

    # save figure
    plt.savefig(save_path, dpi=generic_dpi)
    plt.close()

# ----------------------------------------------------------------------------------------------------------------------
# MISC
# ----------------------------------------------------------------------------------------------------------------------

def get_tick_params(points, num_ticks=5):
    if len(points) > 10:
        step = int(len(points) / num_ticks)
        ticks = np.array([x for x in range(step, len(points), step)])
        ticklabels = points[ticks]
    else:
        ticks = [tick for tick in range(len(points))]
        ticklabels = points
    return ticks, ticklabels

def create_gif(save_dir, image_paths, name='animate.gif'):

    def update(i):
        im.set_array(img_arr[i])
        return im,

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

    save_path = os.path.join(save_dir, name)

    animation_fig.save(save_path, writer="pillow")

    # plt.save(save_path, animation_fig)

    plt.close()
