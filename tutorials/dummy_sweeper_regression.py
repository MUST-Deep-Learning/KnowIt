""" Hyperparameter Tuning (Sweeps)
----------------------------------

This is an example script that demonstrates how a user can perform hyper-
parameter sweeps using Weights and Biases.

When performing a sweep, KnowIt's train method should be provided with an extra
parameter called ``sweep_kwargs', which is a dictionary consisting of the following:
- 'sweep_name': the name of the current sweep
- 'run_name': the name of the current run
- 'log_to_local': whether to log the model from the current run to the local machine.

For more information on configuring the sweep parameters, see the Weights and
Biases documentation:
https://wandb.ai/site/
"""
from markdown_it.rules_inline import entity

import wandb
from knowit import KnowIt

def runner():

    run = wandb.init()
    config = wandb.config

    # current run name
    vals = config.as_dict()
    run_name = ""
    for key in vals:
        run_name += f"{key}_{vals[key]}-"
    run_name = run_name[:-1]
    run.name = run_name

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
                    'max_epochs': 5,
                    'learning_rate': 0.01,
                    'task': 'regression',
                    }

    sweep_kwargs = {'sweep_name': sweep_name, 'run_name': run_name, 'log_to_local': True}

    KI.train_model(
        model_name=model_name,
        kwargs={'data': data_args, 'arch': arch_args, 'trainer': trainer_args},
        sweep_kwargs=sweep_kwargs,
        safe_mode=True,
    )


if __name__ == "__main__":

    # weights and biases variables
    project_name = "my_project_name"
    entity = "my_wb_username_or_team"
    sweep_name = "basic_sweep"
    max_runs = 20
    sweep_method = "bayes"
    # KnowIt variables
    exp_output_dir = "/my/exp/output/dir"
    model_name = "some_sweeped_regression_model"


    KI = KnowIt(custom_exp_dir=exp_output_dir)
    sweep_config = {
        "method": sweep_method,
        "name": sweep_name,
        "metric": {
            "goal": "minimize",
            "name": "valid_loss",
        },
        "parameters": {
            "dropout": {"values": [0.0, 0.2, 0.5]},
            "batch_size": {"values": [64, 128, 256]},
            "learning_rate": {"values": [0.001, 0.01, 0.1]},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity)
    wandb.agent(sweep_id=sweep_id, function=runner, count=max_runs)

    KI.consolidate_sweep(model_name, sweep_name)

