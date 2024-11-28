""" Hyperparameter Tuning (Sweeps)
----------------------------------

This is an example script that demonstrates how a user can perform hyper-
parameter sweeps using Weights and Biases.

When performing a sweep, KnowIt's train method should be provided with an extra
parameter called ``sweep'', which is a dictionary consisting of the following:
- 'name': the name of the current sweep
- 'save_mode': the choice of save mode that KnowIt should use.

There are three save modes:
1) 'none': KnowIt will not save output of any form. The status of the hyper-
parameter sweep can be viewed in Weights and Biases.
2) 'all': KnowIt saves all information about each run in the sweep which
consists of a model checkpoint file, lightning logs, and a model args yaml
file. The sweep can be tracked in Weights and Biases.
3) 'best': Only the best performing model, relative to a loss, is saved. The
sweep can be tracked in Weights and Biases.

For more information on configuring the sweep parameters, see the Weights and
Biases documentation:
https://wandb.ai/site/
"""

import wandb
from knowit import KnowIt

def runner():

    run = wandb.init()
    config = wandb.config

    # current sweep name
    sweep_name = ""
    vals = config.as_dict()
    for key in vals:
        sweep_name += f"{key}_{vals[key]}-"

    run.name = sweep_name

    model_name = "my_new_sweeps"
    data_args = {'name': 'synth_1',
                'task': 'regression',
                'in_components': ['x1', 'x2', 'x3', 'x4'],
                'out_components': ['y1'],
                'in_chunk': [-5, 5],
                'out_chunk': [0, 0],
                'split_portions': [0.6, 0.2, 0.2],
                'batch_size': config.batch_size,
                'split_method': 'instance-random',
                'scaling_tag': 'full'}
    arch_args = {'task': 'regression',
                'name': 'MLP',
                'arch_hps': {'dropout': config.dropout,
                            'width': 512}}
    trainer_args = {'loss_fn': 'mse_loss',
                    'optim': 'Adam',
                    'max_epochs': 3,
                    'learning_rate': 0.01,
                    'task': 'regression',
                    }

    ki.train_model(
        model_name=model_name,
        kwargs={'data': data_args, 'arch': arch_args, 'trainer': trainer_args},
        sweep={
            'name': sweep_name,
            'save_mode': 'all',
        },
        safe_mode=True,
    )

def main():
    sweep_id = wandb.sweep(sweep_config, project="sweep_test")
    wandb.agent(sweep_id=sweep_id, function=runner)

if __name__ == "__main__":
    ki = KnowIt(custom_exp_dir="/home/randler/projects/KnowIt")
    sweep_config = {
        "method": "grid",
        "name": "sweep_test",
        "metric": {
            "goal": "minimize",
            "name": "valid_loss",
        },
        "parameters": {
            "dropout": {"values": [0.4, 0.5]},
            "batch_size": {"values": [64, 128]},
        },
    }

    main()

    ki.consolidate_sweep(path="/home/randler/projects/KnowIt/models/my_new_sweeps")

