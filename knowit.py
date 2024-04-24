__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the main KnowIt module.'

# external imports
import importlib
import os
import copy
import torch
import tempfile
import sys

# internal imports
from env.env_user import temp_exp_dir
from env.env_paths import (ckpt_path, model_output_dir, model_args_path,
                           model_interpretations_dir,
                           model_predictions_dir, default_dataset_dir,
                           custom_dataset_dir, default_archs_dir, custom_arch_dir,
                           custom_dataset_path, dataset_path, arch_path,
                           custom_arch_path, root_exp_dir)
from helpers.logger import get_logger
from helpers.file_dir_procs import (yaml_to_dict, dict_to_yaml, dump_at_path)
from data.base_dataset import BaseDataset
from data.classification_dataset import ClassificationDataset
from data.regression_dataset import RegressionDataset
from trainer.trainer import KITrainer
from trainer.trainer_states import TrainNew, ContinueTraining
from setup.setup_action_args import setup_relevant_args
from setup.select_interpretation_points import get_interpretation_inx
from setup.setup_weighted_cross_entropy import proc_weighted_cross_entropy
from helpers.viz import learning_curves, set_predictions, feature_attribution
from interpret.DLS_Captum import DLS
from interpret.DL_Captum import DeepL
from interpret.IntegratedGrad_Captum import IntegratedGrad
from setup.import_custom_arch import import_custom_arch

logger = get_logger()
logger.setLevel(20)


class KnowIt:
    def __init__(self, custom_exp_dir: str = None, global_safe_mode: bool = True,
                 global_device: str = 'gpu', global_and_viz: bool = True,
                 global_overwrite: bool = False, safe_mode: bool = True,
                 overwrite: bool = False):

        """ KnowIt is the main KnowIt class.
            It keeps track of the current experiment directory
            and operates all submodules according to user arguments. """

        # the experiment directory
        self.exp_output_dir = None
        # a placeholder for a temporary experiment directory (if needed)
        self.temp_exp_dir_obj = None
        # the global (default) safe mode value [whether to protect existing files]
        self.global_safe_mode = global_safe_mode
        # the global (default) overwrite value [whether to overwrite existing files]
        self.global_overwrite = global_overwrite
        # the global (default) device value [what device to perform operations on]
        self.global_device = global_device
        # the global (default) and_viz value [whether to also visualize results]
        self.global_and_viz = global_and_viz

        if custom_exp_dir:
            self.exp_output_dir = root_exp_dir(custom_exp_dir, safe_mode, overwrite)
            logger.info('Custom experiment dir: %s', self.exp_output_dir)
        else:
            self.temp_exp_dir_obj = tempfile.TemporaryDirectory(dir=temp_exp_dir)
            self.exp_output_dir = self.temp_exp_dir_obj.name
            logger.info('Temporary experiment dir: %s', self.exp_output_dir)

    def summarize_dataset(self, name: str):

        dataset_dict = self.available_datasets()
        if name in dataset_dict['custom']:
            data_path = custom_dataset_path(name, self.exp_output_dir)
        elif name in dataset_dict['defaults']:
            data_path = dataset_path(name)
        else:
            logger.error('Unknown dataset name %s. Aborting.', name)
            exit(101)

        datamodule = BaseDataset(data_path)
        summary = {'dataset_name': datamodule.name,
                   'components': datamodule.components,
                   'instances': datamodule.instances,
                   'time_delta': datamodule.time_delta}

        return summary

    def global_args(self, device=None, safe_mode=None, and_viz=None, overwrite=None):

        """ Modify and/or return global arguments according to user arguments. """

        if device is not None:
            self.global_device = device
        if safe_mode is not None:
            self.global_safe_mode = safe_mode
        if and_viz is not None:
            self.global_and_viz = and_viz
        if overwrite is not None:
            self.global_overwrite = overwrite

        return {'global_device': self.global_device,
                'global_and_viz': self.global_and_viz,
                'global_overwrite': self.global_overwrite,
                'global_safe_mode': self.global_safe_mode}

    def import_arch(self, custom_arch_path, safe_mode=None, overwrite=None):
        """ Imports the architecture. """
        if safe_mode is None:
            safe_mode = self.global_safe_mode
        if overwrite is None:
            overwrite = self.global_overwrite
        import_custom_arch(custom_arch_path, self.exp_output_dir, safe_mode, overwrite)

    def import_dataset(self, args: dict, safe_mode=None):
        """ Imports the dataset and returns it as a BaseDataset object.
            It expects keys ('data_import_args', ) in the provided dictionary. """
        if safe_mode is None:
            safe_mode = self.global_safe_mode
        relevant_args = setup_relevant_args(args, required_types=('data_import_args', ))
        data_import_args = relevant_args['data_import_args']
        data_import_args['exp_output_dir'] = self.exp_output_dir
        data_import_args['safe_mode'] = safe_mode
        return BaseDataset.from_path(**data_import_args)

    def available_datasets(self):
        """ Returns a dictionary showing the available datasets. """
        default_datasets = os.listdir(default_dataset_dir)
        custom_datasets = os.listdir(custom_dataset_dir(self.exp_output_dir))
        ret_dict = {'defaults': [], 'custom': []}
        for d in default_datasets:
            if d.endswith('.pickle'):
                ret_dict['defaults'].append(d.split('.pickle')[0])
        for d in custom_datasets:
            if d.endswith('.pickle'):
                ret_dict['custom'].append(d.split('.pickle')[0])
        return ret_dict

    def available_archs(self):
        """ Returns a dictionary showing the available architectures. """
        default_archs = os.listdir(default_archs_dir)
        custom_archs = os.listdir(custom_arch_dir(self.exp_output_dir))
        ret_dict = {'defaults': [], 'custom': []}
        for d in default_archs:
            if d.endswith('.py'):
                ret_dict['defaults'].append(d.split('.py')[0])
        for d in custom_archs:
            if d.endswith('.py'):
                ret_dict['custom'].append(d.split('.py')[0])
        return ret_dict

    def train_model(self, model_name: str, args: dict, device=None,
                    safe_mode=None, and_viz=None, overwrite=None):

        if device is None:
            device = self.global_device
        if safe_mode is None:
            safe_mode = self.global_safe_mode
        if and_viz is None:
            and_viz = self.global_and_viz
        if overwrite is None:
            overwrite = self.global_overwrite

        # check that all relevant args are provided
        relevant_args = setup_relevant_args(args, required_types=('data', 'arch', 'trainer', ))

        # Set up required data modules, Models, and traininer arguments
        datamodule, class_counts = self._get_datamodule(relevant_args['data'])
        model, model_params = self._get_arch_setup(relevant_args['arch'],
                                                    datamodule.in_shape,
                                                    datamodule.out_shape)
        trainer_args = self._get_trainer_setup(relevant_args['trainer'],
                                                device,
                                                class_counts,
                                                model,
                                                model_params,
                                                model_name,
                                               safe_mode,
                                               overwrite)

        # Add dynamically generated data dynamics to relevant args and model_args storage
        data_dynamics = {'in_shape': datamodule.in_shape,
                         'out_shape': datamodule.out_shape,
                         'train_size': datamodule.train_set_size,
                         'valid_size': datamodule.valid_set_size,
                         'eval_size': datamodule.eval_set_size}
        if relevant_args['data']['task'] == 'classification':
            data_dynamics['class_set'] = datamodule.class_set
            data_dynamics['class_count'] = datamodule.class_counts
        relevant_args['data_dynamics'] = data_dynamics
        dict_to_yaml(relevant_args, model_args_path(self.exp_output_dir, model_name, safe_mode, overwrite))

        # Instantiate trainer and begin training
        optional_pl_kwargs = trainer_args.pop('optional_pl_kwargs')
        trainer = KITrainer(state=TrainNew, base_trainer_kwargs=trainer_args,
                            optional_pl_kwargs=optional_pl_kwargs)
        trainer.fit_and_eval(dataloaders=(datamodule.get_dataloader('train'),
                                          datamodule.get_dataloader('valid'),
                                          datamodule.get_dataloader('eval')))

        if and_viz and not trainer_args['mute_logger']:
            learning_curves(self.exp_output_dir, model_name)

    def train_model_further(self, model_name: str, max_epochs: int, device=None,
                    safe_mode=None, and_viz=None, overwrite=None):

        # TODO: WIP = This function should train the desired model to the desired max_epochs without
        #  overwriting old learning metrics

        if device is None:
            device = self.global_device
        if safe_mode is None:
            safe_mode = self.global_safe_mode
        if and_viz is None:
            and_viz = self.global_and_viz
        if overwrite is None:
            overwrite = self.global_overwrite

        relevant_args = yaml_to_dict(model_args_path(self.exp_output_dir, model_name))

        epochs_done = relevant_args['trainer']['max_epochs']
        if epochs_done >= max_epochs:
            logger.error('Desired max_epochs(%s) <= epochs already trained (%s). Aborting.',
                         str(max_epochs), str(epochs_done))
            exit(101)

        relevant_args['trainer']['max_epochs'] = max_epochs
        path_to_ckpt = ckpt_path(self.exp_output_dir, model_name)

        # Set up required data modules, Models, and traininer arguments
        datamodule, class_counts = self._get_datamodule(relevant_args['data'])
        model, model_params = self._get_arch_setup(relevant_args['arch'],
                                                    datamodule.in_shape,
                                                    datamodule.out_shape)
        trainer_args = self._get_trainer_setup(relevant_args['trainer'],
                                                device,
                                                class_counts,
                                                model,
                                                model_params,
                                                model_name,
                                               safe_mode,
                                               overwrite)

        # Add dynamically generated data dynamics to relevant args and model_args storage
        data_dynamics = {'in_shape': datamodule.in_shape,
                         'out_shape': datamodule.out_shape,
                         'train_size': datamodule.train_set_size,
                         'valid_size': datamodule.valid_set_size,
                         'eval_size': datamodule.eval_set_size}
        if relevant_args['data']['task'] == 'classification':
            data_dynamics['class_set'] = datamodule.class_set
            data_dynamics['class_count'] = datamodule.class_counts
        relevant_args['data_dynamics'] = data_dynamics
        dict_to_yaml(relevant_args, model_args_path(self.exp_output_dir, model_name, safe_mode, overwrite))

        # Instantiate trainer and begin training
        optional_pl_kwargs = trainer_args.pop('optional_pl_kwargs')
        # trainer = KITrainer(state=TrainNew, base_trainer_kwargs=trainer_args,
        #                     optional_pl_kwargs=optional_pl_kwargs)
        trainer = KITrainer(state=ContinueTraining, base_trainer_kwargs=trainer_args,
                            optional_pl_kwargs=optional_pl_kwargs, ckpt_file=path_to_ckpt)
        trainer.fit_and_eval(dataloaders=(datamodule.get_dataloader('train'),
                                          datamodule.get_dataloader('valid'),
                                          datamodule.get_dataloader('eval')))

        if and_viz and not trainer_args['mute_logger']:
            learning_curves(self.exp_output_dir, model_name)

    def generate_predictions(self, model_name: str, args: dict, device=None,
                             safe_mode=None, and_viz=None, overwrite=None):

        if device is None:
            device = self.global_device
        if safe_mode is None:
            safe_mode = self.global_safe_mode
        if and_viz is None:
            and_viz = self.global_and_viz
        if overwrite is None:
            overwrite = self.global_overwrite

        relevant_args = setup_relevant_args(args, required_types=('predictor',))

        model_args = yaml_to_dict(model_args_path(self.exp_output_dir, model_name))
        datamodule, class_counts = self._get_datamodule(model_args['data'])
        model, model_params = self._get_arch_setup(model_args['arch'],
                                                   datamodule.in_shape,
                                                   datamodule.out_shape)
        path_to_ckpt = ckpt_path(self.exp_output_dir, model_name)
        pt_model = model(**model_params)
        ckpt = torch.load(f=path_to_ckpt)
        state_dict = ckpt['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key[6:]] = state_dict[key]
            del state_dict[key]
        pt_model.load_state_dict(state_dict)
        pt_model.eval()

        save_dir = model_predictions_dir(self.exp_output_dir, model_name, safe_mode, overwrite)

        dataloader = datamodule.get_dataloader(relevant_args['predictor']['prediction_set'], analysis=True)

        inx_dict = {}
        for batch in enumerate(dataloader):
            x = batch[1]['x']
            y = batch[1]['y']
            s_id = batch[1]['s_id']
            prediction = pt_model(x)

            file_name = relevant_args['predictor']['prediction_set'] + '-' + 'batch_' + str(batch[0]) + '.pickle'
            dump_at_path((s_id, prediction.detach(), y), os.path.join(save_dir, file_name))
            for s in s_id:
                inx_dict[s.item()] = batch[0]

        ist_values = datamodule.get_ist_values(relevant_args['predictor']['prediction_set'])

        file_name = '_' + relevant_args['predictor']['prediction_set'] + '-' + 'ist_inx_dict' + '.pickle'
        dump_at_path((ist_values, inx_dict), os.path.join(save_dir, file_name))

        if and_viz:
            set_predictions(self.exp_output_dir, model_name, relevant_args['predictor']['prediction_set'])

    def interpret_model(self, model_name: str, args: dict, device=None, safe_mode=None, and_viz=None, overwrite=None):

        if device is None:
            device = self.global_device
        if safe_mode is None:
            safe_mode = self.global_safe_mode
        if and_viz is None:
            and_viz = self.global_and_viz
        if overwrite is None:
            overwrite = self.global_overwrite

        relevant_args = setup_relevant_args(args, required_types=('interpreter',))

        interpret_args = copy.deepcopy(relevant_args['interpreter'])

        model_args = yaml_to_dict(model_args_path(self.exp_output_dir, model_name))
        datamodule, class_counts = self._get_datamodule(model_args['data'])
        model, model_params = self._get_arch_setup(model_args['arch'],
                                                   datamodule.in_shape,
                                                   datamodule.out_shape)
        path_to_ckpt = ckpt_path(self.exp_output_dir, model_name)

        interpreter_class = KnowIt._get_interpret_setup(relevant_args['interpreter'])
        interpreter = interpreter_class(model=model,
                                        model_params=model_params,
                                        path_to_ckpt=path_to_ckpt,
                                        datamodule=datamodule,
                                        device=device,
                                        i_data=relevant_args['interpreter']['interpretation_set'],
                                        multiply_by_inputs=relevant_args['interpreter']['multiply_by_inputs'],
                                        seed=relevant_args['interpreter']['seed'])

        i_inx = get_interpretation_inx(relevant_args['interpreter'], model_args,
                                       model_predictions_dir(self.exp_output_dir, model_name))
        attributions = interpreter.interpret(pred_point_id=i_inx)
        interpret_args['i_inx'] = i_inx

        save_name = ''
        for a in interpret_args:
            save_name += str(interpret_args[a]) + '-'
        save_name = save_name[:-1] + '.pickle'

        interpret_args['results'] = attributions

        save_dir = model_interpretations_dir(self.exp_output_dir, model_name, safe_mode, overwrite)
        dump_at_path(interpret_args, os.path.join(save_dir, save_name))

        if and_viz:
            feature_attribution(self.exp_output_dir, model_name, relevant_args['interpreter'])

    def _get_datamodule(self, data_args):
        """ Given the data arguments, this function returns the corresponding data module.
            Note: The dataset is chosen from the custom experiment directory before trying the default directory."""

        dataset_dict = self.available_datasets()
        if data_args['name'] in dataset_dict['custom']:
            data_path = custom_dataset_path(data_args['name'], self.exp_output_dir)
        elif data_args['name'] in dataset_dict['defaults']:
            data_path = dataset_path(data_args['name'])
        else:
            logger.error('Unknown dataset name %s. Aborting.', data_args['name'])
            exit(101)
        data_args['data_path'] = data_path

        if data_args['task'] == 'regression':
            datamodule = RegressionDataset(**data_args)
            class_counts = None
        elif data_args['task'] == 'classification':
            datamodule = ClassificationDataset(**data_args)
            class_counts = datamodule.class_counts
        else:
            logger.error('Unknown task type %s.', data_args['task'])
            exit(101)

        return datamodule, class_counts

    def _get_arch_setup(self, arch_args, in_shape, out_shape):
        """ Given the architecture arguments, this function returns the corresponding Model module.
            Note: The Model is chosen from the custom experiment directory before trying the default directory."""

        def import_class_from_path(path, class_name):

            """ Import class from given path. """

            # Determine module name from path
            module_name = path.replace("/", ".").replace("\\", ".").rstrip(".py")

            # Load the module from the given path
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Import the specific class from the module
            cls = getattr(module, class_name)
            return cls

        arch_dict = self.available_archs()
        if arch_args['name'] in arch_dict['custom']:
            archi_path = custom_arch_path(arch_args['name'], self.exp_output_dir)
        elif arch_args['name'] in arch_dict['defaults']:
            archi_path = arch_path(arch_args['name'])
        else:
            logger.error('Unknown arch name %s. Aborting.', arch_args['name'])
            exit(101)

        model = import_class_from_path(archi_path, 'Model')
        model_params = {"input_dim": in_shape,
                        "output_dim": out_shape,
                        "task_name": arch_args['task']}
        if 'arch_hps' in arch_args:
            for hp in arch_args['arch_hps']:
                model_params[hp] = arch_args['arch_hps'][hp]
        return model, model_params

    def _get_trainer_setup(self, trainer_args, device, class_counts, model,
                           model_params, model_name, safe_mode, overwrite):
        """ This function processes the trainer arguments according to a given set of
            dynamically generated parameters. """

        ret_trainer_args = copy.deepcopy(trainer_args)
        if ret_trainer_args['loss_fn'] == 'weighted_cross_entropy':
            ret_trainer_args['loss_fn'] = proc_weighted_cross_entropy(ret_trainer_args['task'],
                                                                      device,
                                                                      class_counts)

        ret_trainer_args['model'] = model
        ret_trainer_args['model_params'] = model_params
        ret_trainer_args['device'] = device
        ret_trainer_args['out_dir'] = model_output_dir(self.exp_output_dir, model_name, safe_mode, overwrite)
        _ = ret_trainer_args.pop('task')

        return ret_trainer_args

    @staticmethod
    def _get_interpret_setup(interpret_args):

        if interpret_args['interpretation_method'] == 'DeepLiftShap':
            return DLS
        elif interpret_args['interpretation_method'] == 'DeepLift':
            return DeepL
        elif interpret_args['interpretation_method'] == 'IntegratedGradients':
            return IntegratedGrad
        else:
            logger.error('Unknown interpreter %s.',
                         interpret_args['interpretation method'])
            exit(101)