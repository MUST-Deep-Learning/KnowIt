__author__ = 'tiantheunissen@gmail.com'
__description__ = 'Contains the main KnowIt module.'

# external imports
import importlib
import os

# internal imports
from env.env_paths import (ckpt_path, exp_output_dir,
                           model_output_dir, model_args_path,
                           model_interpretations_dir,
                           model_predictions_dir)
from helpers.logger import get_logger
from helpers.read_configs import (yaml_to_dict,
                                  safe_mkdir,
                                  dict_to_yaml,
                                  dump_at_path)
from data.base_dataset import BaseDataset
from data.classification_dataset import ClassificationDataset
from data.regression_dataset import RegressionDataset
from trainer.trainer import KITrainer
from trainer.trainer_states import TrainNew, ContinueTraining, EvaluateOnly
from setup.setup_action_args import setup_relevant_args
from setup.select_interpretation_points import get_interpretation_inx
from setup.setup_weighted_cross_entropy import proc_weighted_cross_entropy
from helpers.viz import learning_curves, set_predictions, feature_attribution
from interpret.DLS_Captum import DLS
from interpret.DL_Captum import DeepL
from interpret.IntegratedGrad_Captum import IntegratedGrad

logger = get_logger()
logger.setLevel(20)

class KnowIt:
    def __init__(self, action, args, safe_mode=True, device='gpu', and_viz=True):

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
            self.train_model(args, and_viz)
        elif action == 'tune':
            logger.error('Hyperparameter tuner not implemented yet.')
            exit(101)
        elif action == 'interpret':
            self.interpret_model(args, and_viz)
        elif action == 'predict':
            self.evaluate_model_predictions(args, and_viz)
        else:
            logger.error('Action %s invalid, or not implemented yet.', action)
            exit(101)

    def import_dataset(self, args):
        import_args = setup_relevant_args(args['import'], 'import', self.safe_mode)
        new_base_dataset = BaseDataset.from_path(**import_args)

    def train_model(self, args, and_viz):
        datamodule, class_counts = KnowIt.get_datamodule(args['data'])
        model, model_params = KnowIt.get_arch_setup(args['arch'],
                                                    datamodule.in_shape,
                                                    datamodule.out_shape)
        trainer_args = KnowIt.get_trainer_setup(args['trainer'],
                                                self.device,
                                                class_counts)

        trainer_args['model'] = model
        trainer_args['model_params'] = model_params
        trainer_args['device'] = self.device
        
        data_dynamics = {'in_shape': datamodule.in_shape,
                         'out_shape': datamodule.out_shape,
                         'train_size': datamodule.train_set_size,
                         'valid_size': datamodule.valid_set_size,
                         'eval_size': datamodule.eval_set_size}
        if args['data']['task'] == 'classification':
            data_dynamics['class_set'] = datamodule.class_set
            data_dynamics['class_count'] = datamodule.class_counts
        args['data_dynamics'] = data_dynamics
        
        if 'optional_pl_kwargs' in trainer_args:
            optional_pl_kwargs = trainer_args.pop('optional_pl_kwargs')
        else:
            optional_pl_kwargs = {}

        trainer_loader = datamodule.get_dataloader('train')
        val_loader = datamodule.get_dataloader('valid')
        eval_loader = datamodule.get_dataloader('eval')
        
        state = trainer_args.pop('state')
        
        if state == 'new':
            trainer_args['out_dir'] = model_output_dir(args['id']['experiment_name'],
                                                   args['id']['model_name'])
            dict_to_yaml(args,
                     model_output_dir(args['id']['experiment_name'],
                                      args['id']['model_name']),
                     'model_args.yaml')
            
            trainer = KITrainer(state=TrainNew, base_trainer_kwargs=trainer_args, optional_pl_kwargs=optional_pl_kwargs)
            trainer.fit_and_eval(dataloaders=(trainer_loader, val_loader, eval_loader))
            
            # do not plot training curves if logging is turned off
            if trainer_args['mute_logger']:
                return
        elif 'continue' in state:
            save_dir = model_output_dir(args['id']['experiment_name'],
                                                   args['id']['model_name'])
            safe_mkdir(save_dir, self.safe_mode, overwrite=False)
            dict_to_yaml(args,
                     save_dir,
                     'model_args.yaml')
            
            trainer_args['out_dir'] = save_dir
            trainer = KITrainer(state=ContinueTraining, base_trainer_kwargs=trainer_args, optional_pl_kwargs=optional_pl_kwargs, ckpt_file=state['continue'])
            trainer.fit_and_eval(dataloaders=(trainer_loader, val_loader, eval_loader))
            
            # do not plot training curves if logging is off
            if trainer_args['mute_logger']:
                return
        elif 'eval' in state:
            # TODO: this currently overwrites the model directory. Is there a way to disregard prevent this?
            save_dir = model_output_dir(args['id']['experiment_name'],
                                                   args['id']['model_name'])
            safe_mkdir(save_dir, safe_mode=True, overwrite=False)
            
            trainer_args['out_dir'] = save_dir
            trainer = KITrainer(state=EvaluateOnly, base_trainer_kwargs=trainer_args, optional_pl_kwargs=optional_pl_kwargs, ckpt_file=state['eval'])
            trainer.fit_and_eval(dataloaders=(trainer_loader, val_loader, eval_loader))
            
            # do not plot training curves
            return

        if and_viz:
            learning_curves(args['id'])

    def interpret_model(self, args, and_viz):

        interpretation_args = setup_relevant_args(args['interpret_args'], 'interpret', self.safe_mode)
        model_args = yaml_to_dict(model_args_path(args['id']['experiment_name'],
                                                   args['id']['model_name']))
        datamodule, class_counts = KnowIt.get_datamodule(model_args['data'])
        model, model_params = KnowIt.get_arch_setup(model_args['arch'],
                                                    datamodule.in_shape,
                                                    datamodule.out_shape)

        interpreter_class = KnowIt.get_interpret_setup(interpretation_args)
        path_to_ckpt = ckpt_path(args['id']['experiment_name'], args['id']['model_name'])
        seed = interpretation_args['seed']
        interpreter = interpreter_class(model=model,
                                        model_params=model_params,
                                        path_to_ckpt=path_to_ckpt,
                                        datamodule=datamodule,
                                        device=self.device,
                                        i_data=interpretation_args['interpretation_set'],
                                        multiply_by_inputs=interpretation_args['multiply_by_inputs'],
                                        seed=seed)

        i_inx = get_interpretation_inx(interpretation_args, model_args)
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

        if and_viz:
            feature_attribution(args['id'], args['interpret_args'])

    def evaluate_model_predictions(self, args, and_viz):

        model_args = yaml_to_dict(model_args_path(args['id']['experiment_name'],
                                                  args['id']['model_name']))
        datamodule, class_counts = KnowIt.get_datamodule(model_args['data'])
        model, model_params = KnowIt.get_arch_setup(model_args['arch'],
                                                    datamodule.in_shape,
                                                    datamodule.out_shape)
        path_to_ckpt = ckpt_path(args['id']['experiment_name'], args['id']['model_name'])

        import torch
        pt_model = model(**model_params)
        ckpt = torch.load(f=path_to_ckpt)
        state_dict = ckpt['state_dict']
        for key in list(state_dict.keys()):
            state_dict[key[6:]] = state_dict[key]
            del state_dict[key]
        pt_model.load_state_dict(state_dict)
        pt_model.eval()

        save_dir = model_predictions_dir(args['id']['experiment_name'],
                                             args['id']['model_name'])
        safe_mkdir(save_dir, self.safe_mode, overwrite=False)

        dataloader = datamodule.get_dataloader(args['predict']['prediction_set'], analysis=True)

        inx_dict = {}
        for batch in enumerate(dataloader):
            x = batch[1]['x']
            y = batch[1]['y']
            s_id = batch[1]['s_id']
            prediction = pt_model(x)

            file_name = args['predict']['prediction_set'] + '-' + 'batch_' + str(batch[0]) + '.pickle'
            dump_at_path((s_id, prediction.detach(), y), os.path.join(save_dir, file_name))
            for s in s_id:
                inx_dict[s.item()] = batch[0]

        ist_values = datamodule.get_ist_values(args['predict']['prediction_set'])

        file_name = '_' + args['predict']['prediction_set'] + '-' + 'ist_inx_dict' + '.pickle'
        dump_at_path((ist_values, inx_dict), os.path.join(save_dir, file_name))

        if and_viz:
            set_predictions(args['id'], args['predict']['prediction_set'])

    @staticmethod
    def get_datamodule(experiment_dict):

        data_args = setup_relevant_args(experiment_dict, 'data')
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
        trainer_args = setup_relevant_args(experiment_dict, 'trainer')
        if trainer_args['loss_fn'] == 'weighted_cross_entropy':
            trainer_args['loss_fn'] = proc_weighted_cross_entropy(trainer_args['task'],
                                                                  device,
                                                                  class_counts)

        return trainer_args

    @staticmethod
    def get_interpret_setup(interpret_args):

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
