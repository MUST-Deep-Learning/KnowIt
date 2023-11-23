__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the ki_setup module.'

# external imports
import importlib

import env.env_user
# internal imports
from env.env_paths import exp_path, model_path, exp_output_path
from helpers.logger import get_logger
from helpers.read_configs import yaml_to_dict, safe_mkdir, safe_copy
from data.base_dataset import BaseDataset
from data.classification_dataset import ClassificationDataset
from data.regression_dataset import RegressionDataset
from trainer.trainer import Trainer
from setup.setup_import_args import setup_import_args
from setup.setup_train_args import setup_trainer_args, setup_data_args
from setup.setup_interpret_args import setup_interpret_args

logger = get_logger()
logger.setLevel(20)

class KnowIt:
    def __init__(self, action, experiment_name,
                 custom_model_name, device='gpu',
                 safe_mode=True):
        """init class from dictionary"""

        self.action = action
        self.device = device
        self.experiment_name = experiment_name
        self.experiment_dict = yaml_to_dict(exp_path(experiment_name))
        self.safe_mode = safe_mode
        self.custom_model_name = custom_model_name

        if action == 'import':
            self.import_dataset()
        if action == 'train':
            safe_mkdir(exp_output_path(self.experiment_name),
                       self.safe_mode, overwrite=True)
            safe_copy(exp_output_path(self.experiment_name),
                      exp_path(experiment_name), self.safe_mode)
            self.train_model()
        if action == 'interpret':
            self.interpret_model()
        if action == 'tune':
            self.tune_model()
        if action == 'analyze':
            self.analyze_data()

    def import_dataset(self):

        new_base_dataset = BaseDataset.from_path(**setup_import_args(self.experiment_dict, self.safe_mode))
        logger.info('New base dataset %s successfully imported.', new_base_dataset.name)

    def train_model(self):
        datamodule, class_counts = KnowIt.get_datamodule(self.experiment_dict)
        model, model_params = KnowIt.get_model_setup(self.experiment_dict,
                                                     datamodule.in_shape,
                                                     datamodule.out_shape)
        trainer_args = setup_trainer_args(self.experiment_dict,
                                          self.device,
                                          class_counts)
        trainer_args['model'] = model
        trainer_args['model_params'] = model_params
        trainer_args['train_device'] = self.device
        trainer_args['model_name'] = self.custom_model_name
        trainer_args['experiment_dir'] = exp_output_path(self.experiment_name)

        trainer = Trainer(**trainer_args)
        trainer_loader = datamodule.get_dataloader('train')
        val_loader = datamodule.get_dataloader('valid')
        eval_loader = datamodule.get_dataloader('eval')

        trainer.fit_model(dataloaders=(trainer_loader, val_loader))
        trainer.evaluate_model(eval_dataloader=(trainer_loader, val_loader, eval_loader))

    def interpret_model(self):
        datamodule, class_counts = KnowIt.get_datamodule(self.experiment_dict)
        model, model_params = KnowIt.get_model_setup(self.experiment_dict,
                                                     datamodule.in_shape,
                                                     datamodule.out_shape)
        interpretation_args = setup_interpret_args(self.experiment_dict)
        interpreter_class = KnowIt.get_interpret_setup(interpretation_args)
        path_to_ckpt = model_path(self.experiment_name, self.custom_model_name)
        interpreter = interpreter_class(model=model,
                          model_params=model_params,
                          path_to_ckpt=path_to_ckpt,
                          datamodule=datamodule,
                          i_data=interpretation_args['interpretation_set'])

        attributions = interpreter.interpret(pred_point_id=(500, 600), num_baselines=1000)
        ping = 0

    def tune_model(self):
        ping = 0

    def analyze_data(self):
        ping = 0

    @staticmethod
    def get_datamodule(experiment_dict):

        data_args = setup_data_args(experiment_dict)

        if experiment_dict['task'] == 'regression':
            datamodule = RegressionDataset(**data_args)
            class_counts = None
        elif experiment_dict['task'] == 'classification':
            datamodule = ClassificationDataset(**data_args)
            class_counts = datamodule.class_counts
        else:
            logger.error('Unknown task type %s.', experiment_dict['task'])
            exit(101)

        return datamodule, class_counts

    @staticmethod
    def get_model_setup(experiment_dict, in_shape, out_shape):
        model = importlib.import_module('archs.' + experiment_dict['arch']).Model
        model_params = {"input_dim": in_shape,
                        "output_dim": out_shape,
                        "task_name": experiment_dict['task']}
        if 'arch_hps' in experiment_dict:
            for hp in experiment_dict['arch_hps']:
                model_params[hp] = experiment_dict['arch_hps'][hp]
        return model, model_params

    @staticmethod
    def get_interpret_setup(interpret_args):

        if interpret_args['interpretation_method'] == 'DeepLiftShap':
            from interpret.DLS_Captum import DLS
            return DLS
        else:
            logger.error('Unknown interpreter %s.',
                         interpret_args['interpretation method'])
            exit(101)


