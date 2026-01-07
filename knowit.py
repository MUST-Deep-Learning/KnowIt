""" This module contains the main class of the toolkit: KnowIt."""

from __future__ import annotations
__copyright__ = 'Copyright (c) 2025 North-West University (NWU), South Africa.'
__licence__ = 'Apache 2.0; see LICENSE file for details.'
__author__ = 'tiantheunissen@gmail.com, randlerabe@gmail.com, potgieterharmen@gmail.com'
__description__ = 'Contains the main KnowIt module.'

# external imports
import importlib
import os
import copy
import torch
import tempfile
import sys
import shutil

from wandb.cli.cli import sweep

# internal imports
from env.env_user import temp_exp_dir
from env.env_paths import (ckpt_path, model_output_dir, model_args_path,
                           model_interpretations_dir,
                           model_predictions_dir, default_archs_dir, custom_arch_dir,
                           arch_path, custom_arch_path, root_exp_dir, arch_name,
                           interpretation_name, model_run_dir, model_sweep_dir,
                           list_available_datasets, data_paths)
from helpers.logger import get_logger
from helpers.file_dir_procs import (yaml_to_dict, safe_dump, safe_copy)
from helpers.viz import (plot_learning_curves, plot_set_predictions, plot_feature_attribution)
from helpers.fetch_torch_mods import get_model_score
from setup.setup_action_args import setup_relevant_args
from setup.select_interpretation_points import get_interpretation_inx
from setup.setup_weighted_cross_entropy import proc_weighted_cross_entropy
from setup.import_custom_arch import import_custom_arch, complies
from data.base_dataset import BaseDataset
from data.prepared_dataset import PreparedDataset
from trainer.trainer import KITrainer
from trainer.trainer_states import TrainNew, EvaluateOnly
from interpret.DLS_Captum import DLS
from interpret.DL_Captum import DeepL
from interpret.IntegratedGrad_Captum import IntegratedGrad

logger = get_logger()
logger.setLevel('WARNING')


class KnowIt:
    """This is the main class that manages the current experiment directory
    and operates all submodules according to user-specified arguments.

    Parameters
    ----------
    custom_exp_dir : str | None, default=None
        Custom directory for the experiment output. If not provided, a temporary directory will be created.
    safe_mode : bool, default=True
        Local safe mode setting for the current experiment directory, determining file protection behavior.
    overwrite : bool, default=False
        Flag indicating whether to overwrite the existing experiment directory if it exists.

    Attributes
    ----------
    exp_output_dir : str | None, default=None
        Stores the directory where experiment outputs will be stored.
    temp_exp_dir_obj : tempfile.TemporaryDirectory | None, default=None
        A placeholder for a temporary experiment directory, if a custom directory is not provided.
    global_safe_mode : bool, default=True
        Global setting for safe mode, which determines whether existing files should be protected.
    global_device : str, default='gpu'
        Global setting for the device to be used for operations (e.g. 'cpu' or 'gpu').
    global_and_viz : bool, default=False
        Global setting for visualization, determining whether results should also be visualized.

    Notes
    -----
    If `custom_exp_dir` is provided, it sets up the experiment directory at the specified location.
    Otherwise, it creates a temporary directory for the experiment.
    """
    exp_output_dir = None
    temp_exp_dir_obj = None
    global_safe_mode = True
    global_device = 'gpu'
    global_and_viz = False
    global_verbose = False

    def __init__(self, custom_exp_dir: str | None = None, *, safe_mode: bool = True, overwrite: bool = False) -> None:
        if custom_exp_dir:
            self.exp_output_dir = root_exp_dir(custom_exp_dir, safe_mode, overwrite)
            logger.info('Custom experiment dir: %s', self.exp_output_dir)
        else:
            self.temp_exp_dir_obj = tempfile.TemporaryDirectory(dir=temp_exp_dir)
            self.exp_output_dir = self.temp_exp_dir_obj.name
            logger.info('Temporary experiment dir: %s', self.exp_output_dir)

    def global_args(self, *, device: str | None = None, safe_mode: bool | None = None,
                    and_viz: bool | None = None, verbose: bool = False) -> dict:
        """Modifies and/or return global arguments according to user arguments.

        This function allows the modification of global settings such as the device,
        safe mode, and visualization settings. It also returns the current values of these settings.

        Parameters
        ----------
        device : str | None, default=None
            If provided, sets the global device to be used for operations (e.g. 'cpu' or 'gpu').
        safe_mode : bool | None, default=None
            If provided, sets the global safe mode value, which determines whether existing files should be protected.
        and_viz : bool | None, default=None
            If provided, sets the global visualization setting, determining whether results should also be visualized.
        verbose : bool, default=False
            If provided, sets the global verbose setting, determining whether messages below warnings should be logged.

        Returns
        -------
        global_args_dict : dict[str, Any]
            A dictionary containing the current global arguments.
            The dictionary includes the following keys:
                - 'global_device' (str): The current device setting for global operations.
                - 'global_and_viz' (bool): The current setting for global visualization.
                - 'global_safe_mode' (bool): The current setting for global safe mode.
                - 'global_verbose' (bool): The current setting for global verbose.
        """
        if device is not None:
            self.global_device = device
        if safe_mode is not None:
            self.global_safe_mode = safe_mode
        if and_viz is not None:
            self.global_and_viz = and_viz
        if verbose is not None:
            self.global_verbose = verbose
            if self.global_verbose:
                logger.setLevel('INFO')
            else:
                logger.setLevel('WARNING')

        global_args_dict = {'global_device': self.global_device,
                            'global_and_viz': self.global_and_viz,
                            'global_safe_mode': self.global_safe_mode}

        return global_args_dict

    def summarize_dataset(self, name: str) -> dict:
        """Summarizes the dataset corresponding to the given name.

        Parameters
        ----------
        name : str
            The name of the dataset to summarize. It can be either a custom dataset or a default dataset.
            Use KnowIt.available_datasets to see options.

        Returns
        -------
        summary : dict[str, Any]
            A dictionary containing the summary of the dataset.
            The dictionary includes the following keys:
                - 'dataset_name' (str): The current device setting for global operations.
                - 'components' (list): The current setting for global visualization.
                - 'instances' (list): The current setting for global safe mode.
                - 'time_delta' (datetime.time): The current setting for global safe mode.

        Notes
        -----
            The function first checks if the dataset name exists in the custom datasets. If not found, it then
            checks in the default datasets. If the dataset name is not found in either, it logs an error and exits.
        """
        meta_path, package_path = data_paths(name, self.exp_output_dir)
        datamodule = BaseDataset(meta_path, package_path)
        summary = {'dataset_name': datamodule.name,
                   'components': datamodule.components,
                   'instances': datamodule.instance_names,
                   'time_delta': datamodule.time_delta}

        return summary

    def import_arch(self, new_arch_path: str, *, safe_mode: bool | None = None) -> None:
        """Imports a new architecture from the specified path.

        This function imports a custom architecture from the given path and stores it
        in the experiment output directory. The operation can be performed in safe mode
        to protect existing files.

        Parameters
        ----------
        new_arch_path : str
            The file path to the new architecture that needs to be imported.
        safe_mode : bool | None, default=None
            If provided, sets the safe mode value for this operation. Safe mode determines
            whether existing files should be protected from being overwritten. If not provided,
            the global safe mode setting will be used.
        """
        if safe_mode is None:
            safe_mode = self.global_safe_mode
        import_custom_arch(new_arch_path, self.exp_output_dir, safe_mode)

    def available_datasets(self) -> dict:
        """Returns a dictionary showing the available datasets for the current instance of KnowIt.

        This function lists all available datasets from the default and custom dataset
        directories. It returns a dictionary with two keys: 'defaults' and 'custom', each
        containing a list of dataset names without file extensions.

        Returns
        -------
        data_dict : dict[str, list]
            A dictionary listing default and custom dataset options.
            The dictionary includes the following keys:
                - 'defaults' (list): The names of available default datasets.
                - 'custom' (list): The names of available custom datasets.

        Notes
        -----
            This is a wrapper for `env.env_paths.list_available_datasets`.
        """
        return list_available_datasets(self.exp_output_dir)

    def available_archs(self) -> dict:
        """Returns a dictionary showing the available architectures for the current instance of KnowIt.

        This function lists all available architectures from the default and custom architecture
        directories. It returns a dictionary with two keys: 'defaults' and 'custom', each containing
        a list of architecture names without file extensions.

        Returns
        --------
        arch_dict : dict[str, list]
            A dictionary listing default and custom architecture options.
            The dictionary includes the following keys:
                - 'defaults' (list): The names of available default architectures.
                - 'custom' (list): The names of available custom architectures.

        Notes
        -----
        The function looks for architecture files with the '.py' extension in both the default
        architecture directory and the custom architecture directory specified by the experiment
        output directory.
        """
        default_archs = os.listdir(default_archs_dir)
        custom_archs = os.listdir(custom_arch_dir(self.exp_output_dir))
        arch_dict = {'defaults': [], 'custom': []}
        for d in default_archs:
            if d.endswith('.py') and d != '__init__.py':
                arch_dict['defaults'].append(d.split('.py')[0])
        for d in custom_archs:
            if d.endswith('.py') and d != '__init__.py':
                arch_dict['custom'].append(d.split('.py')[0])
        return arch_dict

    def import_dataset(self, kwargs: dict, *, safe_mode: bool | None = None) -> BaseDataset:
        """Imports the dataset and returns it as a BaseDataset object.

        This function imports a dataset using the provided arguments and returns it as a
        BaseDataset object. The function expects the dictionary to contain specific keys
        required for data import.

        Parameters
        ----------
        kwargs : dict
            A dictionary containing the arguments for data import. It must include the key
            'data_import'.
        safe_mode : bool | None, default=None
            If provided, sets the safe mode value for this operation. Safe mode determines
            whether existing files should be protected from being overwritten. If not provided,
            the global safe mode setting will be used.

        Returns
        --------
        BaseDataset
            The imported dataset as a BaseDataset object.

        Raises
        -------
        KeyError
            If 'data_import' is not present in the provided dictionary.

        Notes
        -----
        See setup.setup_action_args.py for details on the arguments required in args['data_import'].
        """
        if safe_mode is None:
            safe_mode = self.global_safe_mode
        relevant_args = setup_relevant_args(kwargs, required_types=('data_import', ))
        data_import = relevant_args['data_import']
        data_import['exp_output_dir'] = self.exp_output_dir
        data_import['safe_mode'] = safe_mode
        return BaseDataset.from_raw(**data_import)

    def train_model(self, model_name: str, kwargs: dict, *, device: str | None = None,
                    safe_mode: bool | None = None, and_viz: bool | None = None,
                    sweep_kwargs: dict | None = None,
                    preload: bool = True, num_workers: int = 4) -> None:
        """Trains a model given user arguments.

        This function sets up and trains a model using the provided arguments and configurations.
        It checks and uses global settings for the device, safe mode, and visualization unless
        overridden by the provided arguments.

        Parameters
        ----------
        model_name : str
            The name of the model to be trained.
        kwargs : dict
            A dictionary containing the necessary arguments for setting up the data, architecture,
            and trainer. Expected keys are 'data', 'arch', and 'trainer'.
        device : str | None, default=None
            The device to be used for training. Defaults to the global device setting if not provided.
        safe_mode : bool | None, default=None
            If provided, sets the safe mode value for this operation.
            Defaults to the global safe mode setting if not provided.
        and_viz : bool | None, default=None
            If provided, sets the visualization setting for this operation. Defaults to the global
            visualization setting if not provided.
        sweep_kwargs : dict | None, default=None
            Optional kwargs if a hyperparameter sweep is being performed.
            If provided, must contain kwargs (sweep_name: str, run_name: str, and log_to_local: bool).
        num_workers : int, default = 4
            Sets the number of workers to use for loading the dataset.
        preload : bool, default = False
            Whether to preload the raw relevant instances and slice into memory when sampling feature values.

        Notes
        -----
        See setup.setup_action_args.py for details on the arguments required in args.

        Optional visualization is not done if busy with sweep.

        """
        if device is None:
            device = self.global_device
        if safe_mode is None:
            safe_mode = self.global_safe_mode
        if and_viz is None:
            and_viz = self.global_and_viz

        # check that all relevant args are provided
        relevant_args = setup_relevant_args(kwargs, required_types=('data', 'arch', 'trainer', ))

        # Set up required data modules, Models, and trainner arguments
        datamodule = KnowIt._get_datamodule(self.exp_output_dir, relevant_args['data'])
        model, model_params = KnowIt._get_arch_setup(self.exp_output_dir, self.available_archs(),
                                                     relevant_args['arch'], datamodule.in_shape,
                                                     datamodule.out_shape)
        trainer_args = KnowIt._get_trainer_setup(relevant_args['trainer'], device,
                                                 model, model_params, sweep_kwargs,
                                                 self.exp_output_dir, model_name, safe_mode,
                                                 datamodule)

        # Add dynamically generated data characteristics to relevant args for model_args storage
        relevant_args['data_dynamics'] = KnowIt._get_data_dynamics(datamodule)

        if trainer_args.pop('rescale_logged_output_metrics'):
            trainer_args['output_scaler'] = datamodule.y_scaler

        # Instantiate trainer and begin training
        optional_pl_kwargs = trainer_args.pop('optional_pl_kwargs')
        train_loader = datamodule.get_dataloader('train', preload=preload, num_workers=num_workers)
        optional_pl_kwargs['log_every_n_steps'] = min(len(train_loader.batch_sampler), 50)

        trainer = KITrainer(state=TrainNew, base_trainer_kwargs=trainer_args,
                            optional_pl_kwargs=optional_pl_kwargs)
        trainer.fit(dataloaders=(train_loader,
                                 datamodule.get_dataloader('valid', preload=preload, num_workers=num_workers),
                                 datamodule.get_dataloader('eval', preload=False, num_workers=num_workers)))

        if trainer_args['out_dir'] is not None:
            safe_dump(relevant_args, os.path.join(trainer_args['out_dir'], 'model_args.yaml'), safe_mode)

        if and_viz and not trainer_args['logger_status'] and sweep_kwargs is None:
            plot_learning_curves(self.exp_output_dir, model_name)

    def train_model_from_yaml(self, model_name: str, config_path: str, device: str | None = None,
                    safe_mode: bool | None = None, and_viz: bool | None = None,
                    preload: bool = True, num_workers: int = 4) -> None:
        """Trains a model given a config file.

        This function sets up and trains a model using the provided config file model_args.yaml.

        Parameters
        ----------
        model_name : str
            The name of the model to be trained.
        config_path : str
            The path to the config file (model_args.yaml) containing the necessary arguments for setting up the data,
            architecture, and trainer. The config file should be in YAML format.
            The config file should contain the following keys: 'data', 'arch', and 'trainer'.
        device : str | None, default=None
            The device to be used for training. Defaults to the global device setting if not provided.
        safe_mode : bool | None, default=None
            If provided, sets the safe mode value for this operation.
            Defaults to the global safe mode setting if not provided.
        and_viz : bool | None, default=None
            If provided, sets the visualization setting for this operation. Defaults to the global
            visualization setting if not provided.
        num_workers : int, default = 4
            Sets the number of workers to use for loading the dataset.
        preload : bool, default = False
            Whether to preload the raw relevant instances and slice into memory when sampling feature values.

        Notes
        ---
        This function is a wrapper for the train_model function.

        """

        config_args = yaml_to_dict(config_path)

        self.train_model(model_name=model_name, kwargs=config_args,
                         device=device, safe_mode=safe_mode, and_viz=and_viz,
                         preload=preload, num_workers=num_workers)

    def consolidate_sweep(self, model_name: str, sweep_name: str,
                          selection_by_min: bool = True, safe_mode: bool | None = None,
                          wipe_after: bool = False) -> None:
        """
        Consolidates the results of a hyperparameter sweep by selecting the best run
        and copying its artifacts into the model's main directory.

        This method evaluates all runs within the specified sweep directory, selects the best run based
        on the scoring metric, and safely transfers the relevant files to the model's main directory.
        Optionally, it can wipe the sweep directory after consolidation.

        Parameters
        ----------
            model_name (str): Name of the model associated with the sweep.
            sweep_name (str): Name of the hyperparameter sweep to consolidate.
            selection_by_min (bool, optional): If True, selects the run with the minimum score;
                otherwise, selects the run with the maximum score. Defaults to True.
            safe_mode (bool | None, optional): If True, prevents deletion of files during consolidation.
                If None, defaults to `self.global_safe_mode`. Defaults to None.
            wipe_after (bool, optional): If True, deletes the sweep directory after consolidation.
                Ignored if `safe_mode` is True. Defaults to False.

        Notes
        -----
            - The function assumes the model's output directory structure follows specific conventions.
            - Uses `get_model_score` to evaluate the performance of each run.
            - Safe copy ensures overwriting is controlled by the `safe_mode` flag.
        """

        if safe_mode is None:
            safe_mode = self.global_safe_mode

        logger.info("Consolidating sweep: %s.", sweep_name)
        sweep_dir = model_sweep_dir(self.exp_output_dir, model_name, sweep_name)
        model_dir = model_output_dir(self.exp_output_dir, model_name)

        runs = [r for r in os.listdir(sweep_dir)]
        if len(runs) == 0:
            logger.error("No runs found in sweep: %s.", sweep_name)
            exit(101)

        selected_run = None
        best_score = None
        for r in runs:
            run_score, metric, _ = get_model_score(os.path.join(sweep_dir, r))
            if (best_score is None or (run_score < best_score and selection_by_min) or
                    (run_score > best_score and not selection_by_min)):
                selected_run = r
                best_score = run_score
        selected_run_dir = os.path.join(sweep_dir, selected_run)
        child_content = os.listdir(selected_run_dir)

        for f in os.listdir(model_dir):
            if f.endswith('.ckpt') and not safe_mode:
                os.remove(os.path.join(model_dir, f))
            if f.endswith('.yaml') and not safe_mode:
                os.remove(os.path.join(model_dir, f))
            if f == 'lightning_logs' and not safe_mode:
                shutil.rmtree(os.path.join(model_dir, f))

        for f in child_content:
            safe_copy(path=os.path.join(selected_run_dir, f),
                      new_path=os.path.join(model_dir, f), safe_mode=safe_mode)

        if wipe_after and not safe_mode:
            shutil.rmtree(sweep_dir)

    def run_model_eval(self, model_name: str, device: str | None=None,
                       preload: bool = True, num_workers: int = 4) -> None:
        """Run model evaluation over dataloaders.

        Given a trained model name, evaluates the model on the train, valid-
        ation, and evaluation dataloaders. Prints the evaluation metrics to the
        terminal and saves the metrics in the metrics.csv file in the lightning
        _logs folder.

        Parameters
        ----------
        model_name : str
            The names of the trained model.
        device : str | None
            The device to use for the evaluation (cpu or gpu).
        num_workers : int, default = 4
            Sets the number of workers to use for loading the dataset.
        preload : bool, default = False
            Whether to preload the raw relevant instances and slice into memory when sampling feature values.
        """
        if device is None:
            device = self.global_device

        ckpt_file = ckpt_path(exp_output_dir=self.exp_output_dir, name=model_name)

        trained_model_dict = KnowIt._load_trained_model(self.exp_output_dir,
                                                        self.available_archs(),
                                                        model_name, device, w_pt_model=True)

        trainer_args = trained_model_dict['model_args']['trainer']
        optional_pl_kwargs = trainer_args.pop('optional_pl_kwargs') # empty dictionary
        trainer_args.pop('task')
        trainer_args['model'] = trained_model_dict['model']
        trainer_args['device'] = device
        trainer_args['model_params'] = trained_model_dict['model_params']
        trainer_args['out_dir'] = model_output_dir(self.exp_output_dir, model_name)
        trainer_args['device'] = device

        if trainer_args.pop('rescale_logged_output_metrics'):
            trainer_args['output_scaler'] = trained_model_dict['datamodule'].y_scaler

        trainer = KITrainer(
            state=EvaluateOnly,
            ckpt_file=ckpt_file,
            base_trainer_kwargs=trainer_args,
            optional_pl_kwargs=optional_pl_kwargs,
            train_flag='evaluate_only',
        )

        trainer.evaluate_fitted_model(dataloaders=(
            trained_model_dict['datamodule'].get_dataloader('train', analysis=True, preload=preload, num_workers=num_workers),
            trained_model_dict['datamodule'].get_dataloader('valid', analysis=True, preload=preload, num_workers=num_workers),
            trained_model_dict['datamodule'].get_dataloader('eval', analysis=True, preload=preload, num_workers=num_workers)))

    def generate_predictions(self, model_name: str, kwargs: dict, *, device: str | None = None,
                             safe_mode: bool | None = None, and_viz: bool | None = None) -> None:
        """Generates predictions using a trained model and saves them to disk.

        This method loads a trained model, retrieves a dataloader for the specified prediction set,
        and generates predictions for each batch of data. The predictions, along with the sample IDs
        and ground truth labels, are saved to disk. Additionally, a dictionary mapping sample IDs
        to batch indices is created and saved. Optionally, visualizations can be generated based on
        the predictions.

        Parameters
        ----------
        model_name : str
            The name of the trained model to use for generating predictions.
        kwargs : dict
            A dictionary of arguments required for setting up the prediction process.
            Must include a 'predictor' key with relevant settings.
        device : str | None, default=None
            The device to be used for prediction. Defaults to the global device setting if not provided.
        safe_mode : bool | None, default=None
            Whether to operate in safe mode, which affects how data is saved. If not provided,
            the global safe mode setting is used.
        and_viz : bool | None, default=None
            Whether to generate visualizations based on the predictions. If not provided,
            the global visualization setting is used.

        Notes
        -----
        See setup.setup_action_args.py for details on the arguments required in args['predictor'].
        """
        if device is None:
            device = self.global_device
        if safe_mode is None:
            safe_mode = self.global_safe_mode
        if and_viz is None:
            and_viz = self.global_and_viz

        # get directory to save predictions and set up interpretation args
        save_dir = model_predictions_dir(self.exp_output_dir, model_name, safe_mode)
        relevant_args = setup_relevant_args(kwargs, required_types=('predictor',))

        # get details on trained model and construct appropriate dataloader
        trained_model_dict = KnowIt._load_trained_model(self.exp_output_dir,
                                                        self.available_archs(), model_name, device, w_pt_model=True)
        dataloader = trained_model_dict['datamodule'].get_dataloader(relevant_args['predictor']['prediction_set'],
                                                                     analysis=True)

        if device == 'gpu':
            pred_device = torch.device('cuda')
            trained_model_dict['pt_model'].to(pred_device)
        else:
            pred_device = torch.device('cpu')

        # loop through dataloader get trained model predictions, along with sample id's and batch id's
        # for each batch with trained model.
        inx_dict = {}
        for batch_id, batch in enumerate(dataloader):
            y = batch['y']
            s_id = batch['s_id']
            batch['x'] = batch['x'].to(pred_device)
            y = y.to(pred_device)

            if hasattr(trained_model_dict['pt_model'], 'update_states'):
                trained_model_dict['pt_model'].update_states(batch['ist_idx'][0], batch['x'].device)
            if hasattr(trained_model_dict['pt_model'], 'hard_set_states'):
                trained_model_dict['pt_model'].hard_set_states(batch['ist_idx'][-1])

            prediction = trained_model_dict['pt_model'](batch['x'])

            prediction = prediction.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            if relevant_args['predictor']['rescale_outputs']:
                if trained_model_dict['datamodule'].scaling_tag == 'full':
                    prediction = trained_model_dict['datamodule'].y_scaler.inverse_transform(prediction)
                    y = trained_model_dict['datamodule'].y_scaler.inverse_transform(y)

            file_name = relevant_args['predictor']['prediction_set'] + '-' + 'batch_' + str(batch_id) + '.pickle'
            safe_dump((s_id, prediction, y), os.path.join(save_dir, file_name), safe_mode)
            for s in s_id:
                try:
                    inx_dict[s.item()] = batch_id
                except:
                    for _ in s:
                        inx_dict[_.item()] = batch_id

        # retrieve (instance, slice, and timestep) indices
        ist_values = trained_model_dict['datamodule'].get_ist_values(relevant_args['predictor']['prediction_set'])

        file_name = '_' + relevant_args['predictor']['prediction_set'] + '-' + 'ist_inx_dict' + '.pickle'
        safe_dump((ist_values, inx_dict), os.path.join(save_dir, file_name), safe_mode)

        if and_viz:
            plot_set_predictions(self.exp_output_dir, model_name, relevant_args['predictor']['prediction_set'])

    def interpret_model(self, model_name: str, kwargs: dict, *, device: str | None = None,
                        safe_mode: bool | None = None, and_viz: bool | None = None) -> None:
        """Interpret a trained model and save the interpretation results.

        This method loads a trained model, sets up the interpretation process, and generates interpretations
        for the specified prediction points. The interpretation results are saved to disk.
        Optionally, visualizations based on the interpretation can be generated.

        Parameters
        ----------
        model_name : str
            The name of the trained model to be interpreted.
        kwargs : dict
            A dictionary of arguments required for setting up the interpretation process.
            Must include an 'interpreter' key with relevant settings.
        device : str, default=None
            The device to use for computation (e.g., 'cpu', 'cuda'). If not provided, the global device
            setting is used.
        safe_mode : bool, default=None
            Whether to operate in safe mode, which affects how data is saved. If not provided,
            the global safe mode setting is used.
        and_viz : bool, default=None
            Whether to generate visualizations based on the interpretation. If not provided,
            the global visualization setting is used.

        Notes
        -----
        See setup.setup_action_args.py for details on the arguments required in args['interpreter'].
        """
        if device is None:
            device = self.global_device
        if safe_mode is None:
            safe_mode = self.global_safe_mode
        if and_viz is None:
            and_viz = self.global_and_viz

        relevant_args = setup_relevant_args(kwargs, required_types=('interpreter',))

        interpret_args = copy.deepcopy(relevant_args['interpreter'])

        save_dir = model_interpretations_dir(self.exp_output_dir, model_name, safe_mode)

        trained_model_dict = KnowIt._load_trained_model(self.exp_output_dir,
                                                        self.available_archs(),
                                                        model_name, device, w_pt_model=False)

        # TODO: If the call in the previous line is done with w_pt_model=True,
        #  the actual Pytorch model will also be returned under the key 'pt_model'.
        #  This can then be passed to interpreter_class below.
        #  This means that we don't have to pass (model, model_params, and path_to_ckpt). This will require,
        #  some changes to the interpreter module.

        interpreter_class = KnowIt._get_interpret_setup(relevant_args['interpreter'])
        interpreter = interpreter_class(model=trained_model_dict['model'],
                                        model_params=trained_model_dict['model_params'],
                                        path_to_ckpt=trained_model_dict['path_to_ckpt'],
                                        datamodule=trained_model_dict['datamodule'],
                                        device=device,
                                        i_data=relevant_args['interpreter']['interpretation_set'],
                                        multiply_by_inputs=relevant_args['interpreter']['multiply_by_inputs'],
                                        seed=relevant_args['interpreter']['seed'])

        data_tag = relevant_args['interpreter']['interpretation_set']
        data_selection_matrix = trained_model_dict['datamodule'].selection[
            relevant_args['interpreter']['interpretation_set']]
        i_size = relevant_args['interpreter']['size']
        i_selection_tag = relevant_args['interpreter']['selection']
        seed = relevant_args['interpreter']['seed']
        predictions_dir = model_predictions_dir(self.exp_output_dir, model_name)
        i_inx, predictions, targets, timestamps = get_interpretation_inx(data_tag, data_selection_matrix,
                                                                                 i_size, i_selection_tag,
                                                                                 predictions_dir, seed)

        interpret_args['results'] = interpreter.interpret(pred_point_id=i_inx)
        interpret_args['i_inx'] = i_inx
        interpret_args['input_features'] = trained_model_dict['datamodule'].fetch_input_points_manually(
            interpret_args['interpretation_set'], i_inx)['x']

        interpret_args['input_features'] = interpret_args['input_features'].detach().cpu().numpy()
        if interpret_args['rescale_inputs']:
            if trained_model_dict['datamodule'].scaling_tag in ('full', 'in_only'):
                interpret_args['input_features'] = trained_model_dict['datamodule'].x_scaler.inverse_transform(
                    interpret_args['input_features'])

        if isinstance(i_inx, int):
            interpret_args['targets'] = targets[i_inx]
            interpret_args['predictions'] = predictions[i_inx]
            interpret_args['timestamps'] = timestamps[i_inx]
        else:
            interpret_args['targets'] = [targets[i] for i in i_inx]
            interpret_args['predictions'] = [predictions[i] for i in i_inx]
            interpret_args['timestamps'] = [timestamps[i] for i in i_inx]

        save_name = interpretation_name(interpret_args)
        safe_dump(interpret_args, os.path.join(save_dir, save_name), safe_mode)

        if and_viz:
            plot_feature_attribution(self.exp_output_dir, model_name, save_name)

    def export(self, model_name: str, *, data_tag: str = None, preload: bool = True, num_workers: int = 4,
               device: str | None = None, ret_model: str = None, ret_args: bool = False,
               ret_scaling_args: bool = False) -> dict:

        """
        Load a trained model, prepare specified dataloaders, and return requested components.

        This function facilitates loading of a trained model based on its name, setting up
        specified dataloaders, and optionally returning the model itself, its arguments, or
        the dataloaders based on the input configuration.

        Parameters
        ----------
        model_name : str
            The name of the trained model to be interpreted.
        data_tag : str, default=None
            Specifies the dataloader to return. (e.g, 'train', 'valid', 'test')
        preload : bool, default=True
            Specifies whether to preload the raw relevant instances into memory when sampling
            feature values.
        num_workers : int, default=4
            The number of workers to utilize for dataset loading operations.
        device : str or None, default=None
            The computation device to be used during model loading and execution
            (e.g., 'cuda', 'cpu'). If `None`, the global device is used.
        ret_model : str, default=None
            Indicates whether the trained model should be included in the returned dictionary and returns either
            the lightning or pytorch model (e.g. 'pytorch', 'lightning').
        ret_args : bool, default=False
            Indicates if the model's hyperparameters should be included in the returned dictionary.
        ret_scaling_args: bool, default=False
            Indicates if the 'x_scaler' and 'y_scaler' arguments should be included in the returned dictionary.

        Returns
        -------
        dict
            A dictionary containing the requested components:

            - 'model' (if `ret_model` is True): The trained PyTorch or Lightning model.
            - 'dataloader' (if `data_tag` is one of {'train', 'valid', 'eval'}):
               The specified dataloader for the dataset the model was trained on.
            - 'model_args' (if `ret_args` is True): The hyperparameters or configuration arguments
              of the model.
            - 'scaling_args' (if `ret_scaling_args` is True): The scaling arguments of the model.
        """

        if device is None:
            device = self.global_device

        ret_dict = dict()
        if ret_model is None and data_tag is None and not ret_args and not ret_scaling_args:
            logger.error('Unspecified modules to return.')
            exit(101)

        trained_model_dict = KnowIt._load_trained_model(self.exp_output_dir,
                                                        self.available_archs(), model_name, device, w_pt_model=True)

        if ret_model == 'pytorch':
            ret_dict['model'] = trained_model_dict['pt_model']
        elif ret_model == 'lightning':
            ret_dict['model'] = trained_model_dict['model']
        elif ret_model not in {'pytorch', 'lightning', None}:
            logger.error("Unspecified model to return. Please use one of {'pytorch', 'lightning', None}")
            exit(101)

        if ret_args:
            ret_dict['model_args'] = trained_model_dict['model_args']

        if ret_scaling_args:
            ret_dict['scaling_args'] = {'x_scaler': trained_model_dict['datamodule'].x_scaler,
                                        'y_scaler': trained_model_dict['datamodule'].y_scaler}

        if data_tag in {'train', 'valid', 'eval'}:
            ret_dataloader =  trained_model_dict['datamodule'].get_dataloader(data_tag, analysis=True,
                                                                                    preload=preload,
                                                                                    num_workers=num_workers)
            ret_dict['dataloader'] = ret_dataloader
        elif data_tag not in {'train', 'valid', 'eval', None}:
            logger.error("Please specify the dataloaders as one or more of {'train', 'valid', 'eval', None}")
            exit(101)

        return ret_dict

    @staticmethod
    def _load_trained_model(exp_output_dir: str, available_archs: dict,
                            model_name: str, device, w_pt_model: bool = False) -> dict:
        """Load a trained model along with details on its construction.

        This method loads the configuration, data module, model architecture, and checkpoint path
        for a specified trained model. Optionally, it can also load the actual PyTorch model.

        Parameters
        ----------
        exp_output_dir : str
            The directory containing the experiment outputs.
        available_archs : dict
            A dictionary listing the available model architectures.
        model_name : str
            The name of the trained model to load.
        device : str | None
            The device to use for the evaluation (cpu or gpu).
        w_pt_model : bool, default=False
            Whether to load the actual PyTorch model. If set to True, the PyTorch model is loaded
            and included in the returned dictionary. Default is False.

        Returns
        -------
        dict[str, any]
            A dictionary containing the following keys:
                - 'model_args' (dict): The arguments/configuration used for the model.
                - 'datamodule' (PreparedDataset): The data module associated with the model.
                - 'path_to_ckpt' (str): The path to the model checkpoint.
                - 'model' (type): The untrained PyTorch model.
                - 'model_params' (dict): The model hyperparameters.
                - 'pt_model' (torch.nn.Module, optional): The loaded PyTorch model, included if w_pt_model is True.
        """
        model_args: dict = yaml_to_dict(model_args_path(exp_output_dir, model_name))
        datamodule = KnowIt._get_datamodule(exp_output_dir, model_args['data'])
        model, model_params = KnowIt._get_arch_setup(exp_output_dir, available_archs,
                                                     model_args['arch'], datamodule.in_shape,
                                                     datamodule.out_shape)
        path_to_ckpt = ckpt_path(exp_output_dir, model_name)
        ret_dict = {'model_args': model_args,
                    'datamodule': datamodule,
                    'path_to_ckpt': path_to_ckpt,
                    'model': model,
                    'model_params': model_params}
        if w_pt_model:
            ret_dict['pt_model'] = KnowIt._load_trained_pt_model(model, path_to_ckpt, model_params)

        if ret_dict['model_args']['trainer']['loss_fn'] == 'weighted_cross_entropy':
            if ret_dict['model_args']['trainer']['task'] == 'classification':
                ret_dict['model_args']['trainer']['loss_fn'] = proc_weighted_cross_entropy(datamodule.class_counts,
                                                                                           device)
            else:
                logger.error('weighted_cross_entropy only supported for classification tasks.')
                exit(101)

        return ret_dict

    @staticmethod
    def _load_trained_pt_model(model: torch.nn.Module, path_to_ckpt: str,
                               model_params: dict, to_eval: bool = True) -> torch.nn.Module:
        """Load a trained PyTorch model from a checkpoint.

        This method initializes a PyTorch model with the given arguments, loads the model's state
        dictionary from a checkpoint file, and optionally sets the model to evaluation mode.

        Parameters
        ----------
        model : torch.nn.Module
            The model class or function to initialize the PyTorch model.
        path_to_ckpt : str
            The path to the checkpoint file containing the trained model's state dictionary.
        model_params : dict
            A dictionary of parameters to initialize the PyTorch model.
        to_eval : bool, default=True
            Whether to return the value in eval mode.

        Returns
        -------
        torch.nn.Module
            The loaded PyTorch model.
        """
        pt_model = model(**model_params)
        ckpt = torch.load(f=path_to_ckpt)
        state_dict = ckpt['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key[6:]] = state_dict[key]
            del state_dict[key]
        pt_model.load_state_dict(state_dict)

        if to_eval:
            pt_model.eval()

        return pt_model

    @staticmethod
    def _get_datamodule(exp_output_dir: str, data_args: dict) -> PreparedDataset:
        """Retrieve the appropriate data module based on the provided data arguments.

        This method determines the data path based on whether the dataset is custom or default and
        initializes the appropriate data module.

        Parameters
        ----------
        exp_output_dir : str
            The directory containing the experiment outputs.
        data_args : dict
            A dictionary containing the arguments needed to set up the data module. Must include:
            - 'name': The name of the dataset.
            - 'task': The type of task ('regression' or 'classification').

        Returns
        -------
        PreparedDataset
            The initialized data module.

        Notes
        -----
        The dataset is chosen from the custom experiment directory before trying the default directory.
        """
        meta_path, package_path = data_paths(data_args['name'], exp_output_dir)
        additional_args = {'meta_path': meta_path, 'package_path': package_path}

        if data_args['task'] in ('regression', 'classification', 'vl_regression'):
            datamodule = PreparedDataset(**{**data_args, **additional_args})
        else:
            logger.error('Unknown task type %s.', data_args['task'])
            exit(101)

        return datamodule

    @staticmethod
    def _get_arch_setup(exp_output_dir: str, available_archs: dict, arch_args: dict,
                        in_shape: tuple, out_shape: tuple) -> tuple:
        """Given the architecture arguments, return the corresponding untrained Model module.

        This function identifies and imports the appropriate model architecture based on the provided
        arguments. It first checks the custom experiment directory for the specified model; if not
        found, it checks the default directory. It then sets up the model with the provided input
        and output shapes and additional hyperparameters.

        Parameters
        ----------
        exp_output_dir : str
            The directory containing the experiment outputs.
        available_archs : dict
            A dictionary with keys 'custom' and 'defaults' listing the available architectures.
        arch_args : dict
            The arguments/configuration for the architecture, including the name and hyperparameters.
        in_shape : tuple
            The input shape for the model.
        out_shape : tuple
            The output shape for the model.

        Returns
        -------
        tuple
            A tuple containing:
            - model : class
                The imported Model class.
            - model_params : dict
                A dictionary of parameters to initialize the model, including input and output dimensions
                and task-specific hyperparameters.
        """
        def import_class_from_path(path: str, class_name: str) -> type:
            """Import class from given path.

            Parameters
            ----------
            path : str
                The file path to the module.
            class_name : str
                The name of the class to import.

            Returns
            -------
            type
                The imported class.
            """
            # check that the arch still complies with KnowIt requirements
            if not complies(path):
                logger.error('Selected architecture at %s no longer complies with KnowIt requirements. '
                             'Aborting.', path)
                exit(101)

            # Determine module name from path
            module_name = arch_name(path)

            # Load the module from the given path
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Import the specific class from the module
            cls = getattr(module, class_name)
            return cls

        if arch_args['name'] in available_archs['custom']:
            archi_path = custom_arch_path(arch_args['name'], exp_output_dir)
        elif arch_args['name'] in available_archs['defaults']:
            archi_path = arch_path(arch_args['name'])
        else:
            logger.error('Unknown arch name %s. Aborting.', arch_args['name'])
            exit(101)

        model = import_class_from_path(archi_path, 'Model')
        model_params = {"input_dim": in_shape,
                        "output_dim": out_shape,
                        "task_name": arch_args['task']}
        for hp in arch_args['arch_hps']:
            model_params[hp] = arch_args['arch_hps'][hp]

        return model, model_params

    @staticmethod
    def _get_trainer_setup(trainer_args: dict, device: str, model: type,
                           model_params: dict, sweep_kwargs: dict | None,
                           exp_output_dir: str, model_name: str, safe_mode: bool,
                           datamodule: PreparedDataset) -> dict:
        """Process and return the trainer arguments with dynamically generated parameters.

        This function takes in the initial trainer arguments and modifies them based on
        specific conditions such as the type of loss function. It also incorporates additional
        parameters like the model, device, and save directory into the trainer arguments.

        Parameters
        ----------
        trainer_args : dict
            The initial arguments for the trainer, including configurations such as loss function and task type.
        device : str
            The device on which the model will be trained (e.g., 'cpu' or 'cuda').
        model : type
            The model class to be trained.
        model_params : dict
            The parameters required to initialize the model.
        sweep_kwargs : dict | None, default=None
            Optional kwargs if a hyperparameter sweep is being performed.
            If provided, must contain kwargs (sweep_name: str, run_name: str, and log_to_local: bool).
        exp_output_dir : str
            Path to the current experiment output directory.
        model_name : str
            The name of the model to be trained.
        safe_mode : bool,
            Safe mode value for this operation.
        datamodule : PreparedDataset
            The datamodule object.

        Returns
        -------
        dict
            The processed trainer arguments, including dynamically set parameters. The dictionary contains
                the following keys:
                    - 'loss_fn' (str | dict): The loss function to be used during training,
                        potentially modified for weighted loss.
                    - 'model' (type): The model class to be trained.
                    - 'model_params' (dict): The parameters required to initialize the model.
                    - 'device' (str): The device on which the model will be trained.
                    - 'out_dir' (str): The directory where the training outputs will be saved.
        """
        ret_trainer_args = copy.deepcopy(trainer_args)
        if ret_trainer_args['loss_fn'] == 'weighted_cross_entropy':
            if ret_trainer_args['task'] == 'classification':
                ret_trainer_args['loss_fn'] = proc_weighted_cross_entropy(datamodule.class_counts, device)
            else:
                logger.error('weighted_cross_entropy only supported for classification tasks.')
                exit(101)

        ret_trainer_args['model'] = model
        ret_trainer_args['model_params'] = model_params
        ret_trainer_args['device'] = device
        _ = ret_trainer_args.pop('task')


        if sweep_kwargs is not None and KnowIt._check_sweep_kwargs(sweep_kwargs):
            if sweep_kwargs['log_to_local']:
                ret_trainer_args['logger_status'] = 'w&b_on'
                ret_trainer_args['out_dir'] = model_run_dir(exp_output_dir, model_name,
                                                            sweep_kwargs['sweep_name'], sweep_kwargs['run_name'],
                                                            safe_mode, overwrite=True)
            else:
                ret_trainer_args['logger_status'] = 'w&b_only'
                ret_trainer_args['out_dir'] = None
        else:
            ret_trainer_args['out_dir'] = model_output_dir(exp_output_dir, model_name, safe_mode, overwrite=True)


        return ret_trainer_args

    @staticmethod
    def _get_interpret_setup(interpret_args: dict) -> type:
        """Return the appropriate interpretation class based on the provided interpretation method.

        This function selects and returns the interpretation class corresponding to the
        specified interpretation method in the `interpret_args` dictionary.

        Parameters
        ----------
        interpret_args : dict
            A dictionary containing the arguments for the interpretation setup, specifically the
            `interpretation_method` key which determines which interpretation class to use.

        Returns
        -------
        type
            The class corresponding to the specified interpretation method.

        Raises
        -------
        SystemExit
            If the provided interpretation method is unknown, the function logs an error and exits.

        """
        if interpret_args['interpretation_method'] == 'DeepLiftShap':
            return DLS
        elif interpret_args['interpretation_method'] == 'DeepLift':
            return DeepL
        elif interpret_args['interpretation_method'] == 'IntegratedGradients':
            return IntegratedGrad
        else:
            logger.error('Unknown interpreter %s.',
                         interpret_args['interpretation_method'])
            exit(101)

    @staticmethod
    def _get_data_dynamics(datamodule: PreparedDataset) -> dict:
        """Extract and return dynamic information about the dataset.

        This function gathers various dynamic attributes of the dataset from the provided
        data module and organizes them into a dictionary.

        Parameters
        ----------
        datamodule : object
            The data module object containing the dataset and its properties.

        Returns
        -------
        dict
            A dictionary containing the following keys:
                - 'in_shape' (list): The shape of the input data.
                - 'out_shape' (list): The shape of the output data.
                - 'train_size' (int): The size of the training set.
                - 'valid_size' (int): The size of the validation set.
                - 'eval_size' (int): The size of the evaluation set.
                - 'class_set' (dict): Class name mapping, if task=classification.
                - 'class_counts' (dict): Class count mapping, if task=classification.
        """
        data_dynamics = {'in_shape': datamodule.in_shape,
                         'out_shape': datamodule.out_shape,
                         'train_size': datamodule.train_set_size,
                         'valid_size': datamodule.valid_set_size,
                         'eval_size': datamodule.eval_set_size}

        if datamodule.task == 'classification':
            data_dynamics['class_set'] = datamodule.class_set
            data_dynamics['class_counts'] = datamodule.class_counts

        return data_dynamics

    @staticmethod
    def _check_sweep_kwargs(sweep_kwargs: dict) -> bool:
        for key in ('sweep_name', 'run_name', 'log_to_local'):
            if key not in sweep_kwargs.keys():
                logger.error('Sweep kwarg dictionary must contain %s', key)
                exit(101)
        if type(sweep_kwargs['sweep_name']) != str:
            logger.error('Sweep name must be a string.')
            exit(101)
        if type(sweep_kwargs['run_name']) != str:
            logger.error('Run name must be a string.')
            exit(101)
        if type(sweep_kwargs['log_to_local']) != bool:
            logger.error('Log to local sweep variable must be a boolean.')
            exit(101)
        return True