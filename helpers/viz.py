from env.env_paths import learning_data_path, learning_curves_path
from helpers.read_configs import load_from_csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

train_color = 'rebeccapurple'
valid_color = 'deeppink'
eval_color = 'crimson'
back_color = 'black'
grid_color = 'dimgray'


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

        return result, result_epoch

    curves, num_epochs = get_curves(id_args['experiment_name'], id_args['model_name'])
    result, result_epoch = get_result_epoch(curves)

    loss_curves = [key for key in curves.keys() if 'perf' not in key]
    perf_curves = [key for key in curves.keys() if 'perf' in key]

    if len(perf_curves) > 0:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))

    for c in loss_curves:
        axes[0].plot(curves[c], label=c, marker='.', color=get_color(c))
    axes[0].axvline(x=result_epoch, linestyle='--', c='white')
    check = 0.5 * (axes[0].get_ylim()[1] - axes[0].get_ylim()[0]) + axes[0].get_ylim()[0]
    axes[0].text(result_epoch + 0.1, check, 'result', rotation=90, color='white')
    axes[0].set_xticks([x for x in range(num_epochs)])
    axes[0].set_xticklabels([x + 1 for x in range(num_epochs)])
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_facecolor(back_color)
    axes[0].grid(color=grid_color, alpha=0.5)
    axes[0].legend()

    if len(perf_curves) > 0:
        for c in perf_curves:
            axes[1].plot(curves[c], label=c, marker='.', color=get_color(c))
        axes[1].axvline(x=result_epoch, linestyle='--', c='white')
        check = 0.5 * (axes[1].get_ylim()[1] - axes[1].get_ylim()[0]) + axes[1].get_ylim()[0]
        axes[1].text(result_epoch + 0.1, check, 'result', rotation=90, color='white')
        axes[1].set_xticks([x for x in range(num_epochs)])
        axes[1].set_xticklabels([x + 1 for x in range(num_epochs)])
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Performance metric')
        axes[1].set_facecolor(back_color)
        axes[1].grid(color=grid_color, alpha=0.5)
        axes[1].legend()


    plt.suptitle(str(result), wrap=True)
    # plt.tight_layout()
    # plt.show()

    save_path = learning_curves_path(id_args['experiment_name'], id_args['model_name'])
    plt.savefig(save_path, dpi=200)


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