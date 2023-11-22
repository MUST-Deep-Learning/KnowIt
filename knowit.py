__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the ki_setup module.'

# external imports
import importlib

# internal imports
from env.env_paths import exp_path
from helpers.logger import get_logger
from helpers.read_configs import yaml_to_dict
from data.base_dataset import BaseDataset
from data.classification_dataset import ClassificationDataset
from data.regression_dataset import RegressionDataset
from trainer.trainer import Trainer
from setup.setup_import_args import setup_import_args
from setup.setup_train_args import setup_trainer_args, setup_data_args

logger = get_logger()
logger.setLevel(20)

class KnowIt:
    def __init__(self, action, experiment_name, device='gpu', safe_mode=True):
        """init class from dictionary"""

        self.action = action
        self.device = device
        self.experiment_name = experiment_name
        self.experiment_dict = yaml_to_dict(exp_path(experiment_name))
        self.safe_mode = safe_mode

        if action == 'import':
            self.import_dataset()
        if action == 'train':
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

        data_args = setup_data_args(self.experiment_dict)

        if self.experiment_dict['task'] == 'regression':
            datamodule = RegressionDataset(**data_args)
            class_counts = None
        elif self.experiment_dict['task'] == 'classification':
            datamodule = ClassificationDataset(**data_args)
            class_counts = datamodule.class_counts
        else:
            logger.error('Unknown task type %s.', self.experiment_dict['task'])
            exit(101)

        trainer_args = setup_trainer_args(self.experiment_dict, self.device, class_counts)
        trainer_args['model'] = importlib.import_module('archs.' + self.experiment_dict['arch']).Model
        trainer_args['model_params'] = {"input_dim": datamodule.in_shape,
                                        "output_dim": datamodule.out_shape,
                                        "task_name": self.experiment_dict['task']}
        if 'arch_hps' in self.experiment_dict:
            for hp in self.experiment_dict['arch_hps']:
                trainer_args['model_params'][hp] = self.experiment_dict['arch_hps'][hp]
        trainer_args['experiment_name'] = self.experiment_name
        trainer_args['train_device'] = self.device
        trainer_args['safe_mode'] = self.safe_mode

        trainer = Trainer(**trainer_args)

        trainer_loader = datamodule.get_dataloader('train')
        val_loader = datamodule.get_dataloader('valid')
        eval_loader = datamodule.get_dataloader('eval')

        trainer.fit_model(dataloaders=(trainer_loader, val_loader))
        trainer.evaluate_model(eval_dataloader=(trainer_loader, val_loader, eval_loader))

    def interpret_model(self):

        data_args = setup_data_args(self.experiment_dict)

        if self.experiment_dict['task'] == 'regression':
            datamodule = RegressionDataset(**data_args)
            class_counts = None
        elif self.experiment_dict['task'] == 'classification':
            datamodule = ClassificationDataset(**data_args)
            class_counts = datamodule.class_counts
        else:
            logger.error('Unknown task type %s.', self.experiment_dict['task'])
            exit(101)

        model = importlib.import_module('archs.' + self.experiment_dict['arch']).Model
        model_params = {"input_dim": datamodule.in_shape,
                        "output_dim": datamodule.out_shape,
                        "task_name": self.experiment_dict['task']}

        from interpret.DLS_Captum import DLS
        path_to_ckpt = "/home/tian/projects/KnowIt/default_experiment/models/Model_2023-11-22 17:23:00/bestmodel-epoch=5-val_loss=0.57 2023-11-22 17:23:00.ckpt"
        dls = DLS(model=model,
                  model_params=model_params,
                  path_to_ckpt=path_to_ckpt,
                  datamodule=datamodule,
                  i_data='train')

        attributions = dls.interpret(pred_point_id=(10000, 10100), num_baselines=1000)
        print(attributions[10033][(0, 1)]['attributions'].shape)
        print(attributions[10033][(0, 1)]['attributions'])

    def tune_model(self):
        ping = 0

    def analyze_data(self):
        ping = 0


