import argparse
import yaml
import copy
from pathlib import Path
import shutil
from collections import namedtuple
import sys
import wandb

from scvm.auto.registry import Registry, ParamClsSetting
from scvm.solvers.thp import THP

def create_parser():
    return argparse.ArgumentParser(allow_abbrev=False)


class Config:
    def __init__(self, param_dict):
        self.param_dict = copy.copy(param_dict)


    def has_same_params(self, other):
        return self.param_dict == other.param_dict


    def __getitem__(self, k):
        return self.param_dict[k]


    def __contains__(self, k):
        return k in self.param_dict


    def __setitem__(self, k, v):
        self.param_dict[k] = v


    def get(self, k, default_v):
        return self.param_dict.get(k, default_v)


    def __repr__(self):
        return str(self.param_dict)


    @staticmethod
    def from_yaml(yaml_path):
        return Config(yaml.safe_load(open(yaml_path, 'r')))


    def save_yaml(self, yaml_path):
        open(yaml_path, 'w').write(yaml.dump(self.param_dict))


class ConfigBlueprint:
    def __init__(self, default_param_dict):
        '''
        Args:
          default_param_dict:
            A dict, where the values have to be one of [string, int, float].
        '''
        self.default_param_dict = default_param_dict


    def prepare_parser(self, parser):
        def str2bool(v):
            if isinstance(v, bool):
                return v
            if v.lower() in ('yes', 'true', 't', 'y', '1'):
                return True
            elif v.lower() in ('no', 'false', 'f', 'n', '0'):
                return False
            else:
                raise argparse.ArgumentTypeError(
                    f'Boolean value expected but received {v}.')

        def str2list(s):
            from ast import literal_eval
            return [int(x) for x in s.split(',')]

        for k, v in self.default_param_dict.items():
            if type(v) == bool:
                parser.add_argument('--{}'.format(k), type=str2bool, default=v)
            elif type(v) == list:
                if type(v[0]) == int:
                    parser.add_argument('--{}'.format(k), type=str2list, default=v)
                else:
                    raise argparse.ArgumentTypeError(
                        f'Only support integer list but {k} has default {v}')
            else:
                parser.add_argument('--{}'.format(k), type=type(v), default=v)


ECParseResult = namedtuple('ECParseResult',
                           ['tmp_dict', 'config', 'exp_dir'])


class ExperimentCoordinator:
    def __init__(self, root_dir):
        '''
        We assume the following hierarchy of directories:
          root_dir/exps/exp_name:
            conf.yml:
              The configuration corresponding to an instance of Config class.
          Then arbitrary files and subfolders can be placed here depending
          on the solver.

        This class maintains multiple blueprints that will be combined and
        form an argparser. Hence duplicated keys need to be avoided.

        Args:
            root_dir:
              Root directory of the experiments.
        '''
        self.root_path = Path(root_dir)

        # Temporary blueprints are non-persistent.
        self.temporary_blueprints = []

        # Common blueprints contain common parameters and the ones related
        # to problem.
        self.common_blueprints = [ConfigBlueprint({
            'project': 'uncategorized',
            'wandb': True,
            'seed': 42,
        })]


    def add_temporary_arguments(self, param_dict):
        self.temporary_blueprints.append(ConfigBlueprint(param_dict))


    def add_common_arguments(self, param_dict):
        self.common_blueprints.append(ConfigBlueprint(param_dict))


    def _parse_single_str(self, key, args, required=False):
        parser = create_parser()
        parser.add_argument('--{}'.format(key), type=str, required=required)
        parsed, rest_args = parser.parse_known_args(args)
        return vars(parsed)[key], rest_args


    def parse_args(self):
        '''
        There are two types of arguments: temporary and persistent.
        Temporary arguments won't be saved (e.g. --num_train_step), while
        persistent arguments (e.g. model architecture) will be saved.

        "exp_name" is the name of the experiment which is the same as the
        folder name containing this experiment's related files. If not
        provided, a random unique name will be generated (which can later
        be changed).
        '''
        parser = create_parser()
        rest_args = sys.argv

        exp_name, rest_args = self._parse_single_str('exp_name', rest_args)

        # Load config if it exists.
        exist_config = False
        if exp_name:
            config_path = self._get_conf_yaml(self._get_exps_path() /
                                              exp_name)
            if config_path.exists():
                exist_config = True
                config_dict = Config.from_yaml(config_path).param_dict
                print(f'Found existing config at {config_path}')

        if not exist_config:
            config_dict = {}

        # If override, then skip [Y/N] prompts.
        parser.add_argument('--override', action='store_true', default=False)
        for b in self.temporary_blueprints:
            b.prepare_parser(parser)
        tmp_args, rest_args = parser.parse_known_args(rest_args)
        tmp_dict = vars(tmp_args)
        tmp_dict['exp_name'] = exp_name


        parser = create_parser()
        for b in self.common_blueprints:
            b.prepare_parser(parser)
        common_args, rest_args = parser.parse_known_args(rest_args)
        config_dict.update(vars(common_args))

        # Handle solver arguments.
        solver, rest_args = self._parse_single_str('solver', rest_args)
        if solver is None:
            if 'solver' not in config_dict:
                raise Exception('If omitting --solver, then it'
                                ' must be set in the config.')
            solver = config_dict['solver']
        config_dict['solver'] = solver
        solver_setting = Registry.get_solver_setting(solver)
        solver_blueprint = ConfigBlueprint(
            solver_setting.param_dict
        )
        parser = create_parser()
        solver_blueprint.prepare_parser(parser)
        solver_args, rest_args = parser.parse_known_args(rest_args)
        config_dict['solver_config'] = vars(solver_args)

        # TODO: refactor the need_* code to something cleaner.
        if solver_setting.need_thp:
            # Handle THP.
            parser = create_parser()
            thp_blueprint = ConfigBlueprint(
                Registry.get_thp_param_dict()
            )
            thp_blueprint.prepare_parser(parser)
            thp_args, rest_args = parser.parse_known_args(rest_args)
            tmp_dict['thp'] = vars(thp_args)


        if solver_setting.need_flow:
            # Handle flow arguments.
            flow, rest_args = self._parse_single_str('flow', rest_args)
            if flow is None:
                if 'flow' not in config_dict:
                    raise Exception('If omitting --flow, then it'
                                    ' must be set in the config.')
                flow = config_dict['flow']
            config_dict['flow'] = flow
            flow_blueprint = ConfigBlueprint(
                Registry.get_model_setting(flow).param_dict
            )
            parser = create_parser()
            flow_blueprint.prepare_parser(parser)
            flow_args, rest_args = parser.parse_known_args(rest_args)
            config_dict['flow_config'] = vars(flow_args)

        if solver_setting.need_score:
            # Handle score arguments.
            score, rest_args = self._parse_single_str('score', rest_args)
            if score is None:
                if 'score' not in config_dict:
                    raise Exception('If omitting --score, then it'
                                    ' must be set in the config.')
                score = config_dict['score']
            config_dict['score'] = score
            score_blueprint = ConfigBlueprint(
                Registry.get_model_setting(score).param_dict
            )
            parser = create_parser()
            score_blueprint.prepare_parser(parser)
            score_args, rest_args = parser.parse_known_args(rest_args)
            config_dict['score_config'] = vars(score_args)

        if solver_setting.need_potential:
            # Handle potential arguments.
            potential, rest_args = self._parse_single_str('potential', rest_args)
            if potential is None:
                if 'potential' not in config_dict:
                    raise Exception('If omitting --potential, then it'
                                    ' must be set in the config.')
                potential = config_dict['potential']
            config_dict['potential'] = potential
            potential_blueprint = ConfigBlueprint(
                Registry.get_model_setting(potential).param_dict
            )
            parser = create_parser()
            potential_blueprint.prepare_parser(parser)
            potential_args, rest_args = parser.parse_known_args(rest_args)
            config_dict['potential_config'] = vars(potential_args)

        if solver_setting.need_optimizer:
            # Handle optimizer arguments.
            optimizer, rest_args = self._parse_single_str('optimizer',
                                                          rest_args)
            if optimizer is None:
                if 'optimizer' not in config_dict:
                    raise Exception('If omitting --optimizer, then it'
                                    ' must be set in the config.')
                optimizer = config_dict['optimizer']
            config_dict['optimizer'] = optimizer
            optimizer_blueprint = ConfigBlueprint(
                Registry.get_optimizer_setting(optimizer).param_dict
            )
            parser = create_parser()
            optimizer_blueprint.prepare_parser(parser)
            optimizer_args, rest_args = parser.parse_known_args(rest_args)
            config_dict['optimizer_config'] = vars(optimizer_args)

            # Optionally handle scheduler arguments.
            scheduler, rest_args = self._parse_single_str('scheduler',
                                                          rest_args)
            if scheduler is not None:
                config_dict['scheduler'] = scheduler
                if isinstance(Registry.get_scheduler_setting(scheduler),
                              ParamClsSetting):
                    scheduler_blueprint = ConfigBlueprint(
                        Registry.get_scheduler_setting(scheduler).param_dict
                    )
                    parser = create_parser()
                    scheduler_blueprint.prepare_parser(parser)
                    scheduler_args, rest_args = parser.parse_known_args(rest_args)
                    config_dict['scheduler_config'] = vars(scheduler_args)

        config = Config(config_dict)
        exp_dir = self._make_persistent(config, exp_name,
                                       override=tmp_dict['override'])

        rest_args = rest_args[1:] # first one is always run.py
        if len(rest_args) > 0:
           print('WARNING: Args unprocessed: ', rest_args)

        self.parse_result = ECParseResult(
            tmp_dict=tmp_dict,
            config=config,
            exp_dir=exp_dir
        )
        return self.parse_result


    def _get_exps_path(self):
        path = self.root_path / 'exps/'
        path.mkdir(exist_ok=True)
        return path


    def _get_conf_yaml(self, exp_dir):
        return Path(exp_dir) / 'conf.yml'


    def _make_persistent(self, config, exp_name, override):
        exist = False

        if exp_name is not None:
            exp_dir = self._get_exps_path() / exp_name
            config_path = self._get_conf_yaml(exp_dir)
            if config_path.exists():
                old_config = Config.from_yaml(config_path)
                print(f'Found existing experiment {exp_name}!')
                diff = False
                for k, v in config.param_dict.items():
                    if k not in old_config.param_dict:
                        print(f'Existing config missing {k}!')
                        diff = True
                    elif old_config[k] != v:
                        print(f'Existing config has {k}={old_config[k]}'
                              f' whereas new config has {k}={v}!')
                        diff = True
                for k in old_config.param_dict:
                    if k not in config.param_dict:
                        print(f'New config missing {k}!')
                        diff = True

                if diff and not override:
                    override = input("Override? [Y/N]")
                if override == True or override == 'Y':
                    print('Removing {}...'.format(exp_dir))
                    shutil.rmtree(exp_dir)
                elif diff:
                    raise Exception('Found config with same name'
                                    ' but different parameters! Abort.')
                else:
                    print('Resuming experiment {} with '.format(exp_name) +
                          'identical config...')
                    exist = True
                    config = old_config

        if not exist:
            # Save config
            config['wandb_id'] = wandb.util.generate_id()
            if exp_name is None:
                exp_name = config['wandb_id']
            exp_dir = self._get_exps_path() / exp_name
            exp_dir.mkdir(parents=True, exist_ok=True)
            config.save_yaml(self._get_conf_yaml(exp_dir))
            print('Saved a new config to {}.'.format(exp_dir))

        return exp_dir


    def create_solver(self, problem):
        config = self.parse_result.config
        exp_dir = self.parse_result.exp_dir
        tmp_dict = self.parse_result.tmp_dict

        wandb.init(
            project=config['project'],
            mode='online' if config['wandb'] else 'offline',
            config={
                'exp_dir': exp_dir,
                'tmp_dict': tmp_dict,
                **config.param_dict,
            },
            name=('' if tmp_dict['exp_name'] is None
                  else f'{tmp_dict["exp_name"]}'),
            id=config['wandb_id'],
            resume='allow'
        )

        solver_setting = Registry.get_solver_setting(
            config['solver'])
        solver_cls = solver_setting.cls
        extra = {}
        if solver_setting.need_thp:
            extra['thp'] = THP(**tmp_dict['thp'])
        if solver_setting.need_flow:
            flow_setting = Registry.get_model_setting(
                config['flow'])
            extra['flow'] = flow_setting.cls(
                dim=problem.get_dim(), # always stick in "dim"
                **config['flow_config'])
        if solver_setting.need_score:
            score_setting = Registry.get_model_setting(
                config['score'])
            extra['score'] = score_setting.cls(
                dim=problem.get_dim(), # always stick in "dim"
                **config['score_config'])
        if solver_setting.need_potential:
            potential_setting = Registry.get_model_setting(
                config['potential'])
            extra['potential'] = potential_setting.cls(
                **config['potential_config'])
        if solver_setting.need_optimizer:
            optimizer_setting = Registry.get_optimizer_setting(
                config['optimizer'])
            optimizer_params = config['optimizer_config']
            if 'scheduler' in config:
                scheduler_setting = Registry.get_scheduler_setting(
                    config['scheduler']
                )
                if isinstance(scheduler_setting, ParamClsSetting):
                    optimizer_params['learning_rate'] = scheduler_setting.cls(
                        **config['scheduler_config']
                    )
                else:
                    # Otherwise it's a scheduler.
                    optimizer_params['learning_rate'] = scheduler_setting
            extra['optimizer'] = optimizer_setting.cls(
                **optimizer_params
            )
        solver = solver_cls(problem=problem,
                            exp_dir=exp_dir,
                            seed=config['seed'],
                            **config['solver_config'],
                            **extra)
        return solver

