__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the ki_setup module.'

# external imports
import importlib
import os

import env.env_user
# internal imports
from env.env_paths import (exp_path, ckpt_path, exp_output_dir,
                           model_output_dir, model_args_path, model_interpretations_dir)
from helpers.logger import get_logger
from helpers.read_configs import (yaml_to_dict, safe_mkdir, safe_copy, dict_to_yaml,
                                  dump_at_path, load_from_path)
from data.base_dataset import BaseDataset
from data.classification_dataset import ClassificationDataset
from data.regression_dataset import RegressionDataset
from trainer.trainer import Trainer
from setup.setup_import_args import setup_import_args
from setup.setup_train_args import setup_trainer_args, setup_data_args
from setup.setup_interpret_args import setup_interpret_args
from helpers.viz import learning_curves

logger = get_logger()
logger.setLevel(20)

class KnowIt:
    def __init__(self, action, args, safe_mode=True, device='gpu'):

        self.safe_mode = safe_mode
        self.device = device

        if 'id' in args:
            if 'experiment_name' in args['id']:
                safe_mkdir(exp_output_dir(args['id']['experiment_name']), safe_mode, overwrite=False)
            if 'model_name' in args['id']:
                safe_mkdir(model_output_dir(args['id']['experiment_name'],
                                            args['id']['model_name']), safe_mode, overwrite=action == 'train')

        if action == 'import':
            self.import_dataset(args)
        elif action == 'analyze':
            logger.error('Data analyzer not implemented yet.')
            exit(101)
        elif action == 'train':
            self.train_model(args)
        elif action == 'tune':
            logger.error('Hyperparameter tuner not implemented yet.')
            exit(101)
        elif action == 'interpret':
            self.interpret_model(args)
        else:
            logger.error('Action %s invalid, or not implemented yet.', action)
            exit(101)

    def import_dataset(self, args):
        args['safe_mode'] = self.safe_mode
        new_base_dataset = BaseDataset.from_path(**args['import'])
        # logger.info('New base dataset %s successfully imported.', new_base_dataset.name)

    def train_model(self, args, and_viz=True):
        datamodule, class_counts = KnowIt.get_datamodule(args['data'])
        model, model_params = KnowIt.get_arch_setup(args['arch'],
                                                    datamodule.in_shape,
                                                    datamodule.out_shape)
        # args['trainer']['task'] = args['data']['task']
        trainer_args = KnowIt.get_trainer_setup(args['trainer'],
                                                self.device,
                                                class_counts)

        trainer_args['model'] = model
        trainer_args['model_params'] = model_params
        trainer_args['train_device'] = self.device
        trainer_args['out_dir'] = model_output_dir(args['id']['experiment_name'],
                                                   args['id']['model_name'])
        dict_to_yaml(args,
                     model_output_dir(args['id']['experiment_name'],
                                      args['id']['model_name']),
                     'model_args.yaml')

        trainer = Trainer(**trainer_args)
        trainer_loader = datamodule.get_dataloader('train')
        val_loader = datamodule.get_dataloader('valid')
        eval_loader = datamodule.get_dataloader('eval')

        trainer.fit_model(dataloaders=(trainer_loader, val_loader))
        trainer.evaluate_model(eval_dataloader=(trainer_loader, val_loader, eval_loader))
        if and_viz:
            learning_curves(args['id'])

    def interpret_model(self, args):

        interpretation_args = setup_interpret_args(args['interpret_args'])
        model_args = yaml_to_dict(model_args_path(args['id']['experiment_name'],
                                                   args['id']['model_name']))
        datamodule, class_counts = KnowIt.get_datamodule(model_args['data'])
        model, model_params = KnowIt.get_arch_setup(model_args['arch'],
                                                    datamodule.in_shape,
                                                    datamodule.out_shape)

        interpreter_class = KnowIt.get_interpret_setup(interpretation_args)
        path_to_ckpt = ckpt_path(args['id']['experiment_name'], args['id']['model_name'])
        interpreter = interpreter_class(model=model,
                                        model_params=model_params,
                                        path_to_ckpt=path_to_ckpt,
                                        datamodule=datamodule,
                                        i_data=interpretation_args['interpretation_set'])

        i_inx = KnowIt.get_interpretation_inx(interpretation_args, datamodule)
        attributions = interpreter.interpret(pred_point_id=i_inx)
        interpretation_args['i_inx'] = i_inx

        save_name = ''
        for a in interpretation_args:
            save_name += str(interpretation_args[a]) + '-'
        save_name = save_name[:-1] + '.pickle'

        interpretation_args['results'] = attributions

        save_dir = model_interpretations_dir(args['id']['experiment_name'],
                                                   args['id']['model_name'])
        safe_mkdir(save_dir, self.safe_mode, overwrite=False)
        save_path = os.path.join(save_dir, save_name)
        dump_at_path(interpretation_args, save_path)

    @staticmethod
    def get_datamodule(experiment_dict):

        data_args = setup_data_args(experiment_dict)
        if 'seed' not in data_args:
            data_args['seed'] = 123

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
    def get_arch_setup(experiment_dict, in_shape, out_shape):
        model = importlib.import_module('archs.' + experiment_dict['arch']).Model
        model_params = {"input_dim": in_shape,
                        "output_dim": out_shape,
                        "task_name": experiment_dict['task']}
        if 'arch_hps' in experiment_dict:
            for hp in experiment_dict['arch_hps']:
                model_params[hp] = experiment_dict['arch_hps'][hp]
        return model, model_params

    @staticmethod
    def get_trainer_setup(experiment_dict, device, class_counts):
        trainer_args = setup_trainer_args(experiment_dict,
                                          device,
                                          class_counts)

        if 'seed' not in trainer_args:
            trainer_args['set_seed'] = 123
        else:
            trainer_args['set_seed'] = trainer_args.pop('seed')

        return trainer_args

    @staticmethod
    def get_interpret_setup(interpret_args):

        if interpret_args['interpretation_method'] == 'DeepLiftShap':
            from interpret.DLS_Captum import DLS
            return DLS
        else:
            logger.error('Unknown interpreter %s.',
                         interpret_args['interpretation method'])
            exit(101)

    @staticmethod
    def get_interpretation_inx(interpretation_args, datamodule):

        inx = 0

        if interpretation_args['interpretation_set'] == 'train':
            set_size = datamodule.train_set_size
        elif interpretation_args['interpretation_set'] == 'valid':
            set_size = datamodule.valid_set_size
        elif interpretation_args['interpretation_set'] == 'eval':
            set_size = datamodule.eval_set_size
        else:
            logger.error('Unknown interpretation_set %s', interpretation_args['interpretation_set'])
            exit(101)

        size = interpretation_args['size']
        if set_size < size:
            logger.error('Size of desired interpretation %s larger than desired set %s.',
                         size,
                         interpretation_args['interpretation_set'])
            exit(101)

        if interpretation_args['selection'] == 'random':
            import numpy as np
            start = np.random.randint(0, set_size - size)
            inx = (start, start + size)
        elif interpretation_args['selection'] == 'all':
            inx = (0, set_size)
        elif interpretation_args['selection'] == 'success':
            # TODO: Select chunk where the model did the best.
            logger.warning('Success selection not implemented yet. Reverting to random.')
            import numpy as np
            start = np.random.randint(0, set_size - size)
            inx = (start, start + size)
        elif interpretation_args['selection'] == 'failure':
            # TODO: Select chunk where the model did the best.
            logger.warning('Failure selection not implemented yet. Reverting to random.')
            import numpy as np
            start = np.random.randint(0, set_size - size)
            inx = (start, start + size)
        else:
            logger.error('Invalid interpretation selection %s.', interpretation_args['selection'])
            exit(101)

        return inx