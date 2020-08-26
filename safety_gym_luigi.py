import luigi
import os.path as osp
import subprocess

import itertools
import json
import random

env_type = ['safety_gym']
reward_shaping = [0, -0.05, -0.1, -1]  #, -0.001, -0.0025, -0.005, -0.01, -0.1]
augmentation = ['', 'constraint_state']  #, 'constraint_state_noembed']
train_seed = [7842, 1206, 8610]  #, 5783, 9713]
eval_seed = [5722, 2579, 1892, 7583, 9238]


def twod_tasks():
    env = ['Seaquest']
    constraint = [
        '2d_actuation4', '2d_actuation4_dense', '2d_actuation4_hard',
        '2d_dithering4', '2d_dithering4_dense', '2d_dithering4_hard',
        '2d_actuation4_hardtrain', '2d_actuation4_hardeval',
        '2d_dithering4_hardtrain', '2d_dithering4_hardeval'
    ]
    arg_names = [
        'env_type', 'env', 'constraint', 'reward_shaping', 'augmentation',
        'train_seed', 'eval_seed'
    ]
    args = [
        dict(zip(arg_names, arg_vals)) for arg_vals in itertools.product(
            env_type, env, constraint, reward_shaping, augmentation,
            train_seed, eval_seed)
    ]
    args = [
        d for d in args
        if not ('hard' in d['constraint'] and d['reward_shaping'] != 0)
    ]
    return [EvalTask(**a) for a in args]


def oned_tasks():
    # per-task info
    env = ['Breakout', 'SpaceInvaders']
    constraint = [
        '1d_dithering2', '1d_dithering2_dense', '1d_dithering2_hard',
        '1d_actuation4', '1d_actuation4_dense', '1d_actuation4_hard',
        '1d_dithering2_hardtrain', '1d_dithering2_hardeval',
        '1d_actuation4_hardtrain', '1d_actuation4_hardeval'
    ]
    arg_names = [
        'env_type', 'env', 'constraint', 'reward_shaping', 'augmentation',
        'train_seed', 'eval_seed'
    ]

    args = [
        dict(zip(arg_names, arg_vals)) for arg_vals in itertools.product(
            env_type, env, constraint, reward_shaping, augmentation,
            train_seed, eval_seed)
    ]
    args = [
        d for d in args
        if not ('hard' in d['constraint'] and d['reward_shaping'] != 0)
    ]
    return [EvalTask(**a) for a in args]


def breakout_tasks():
    # per-task info
    env_type = ['atari']
    env = ['Breakout']
    constraint = [
        'paddle_ball_distance', 'paddle_ball_distance_dense',
        'paddle_ball_distance_hard', 'paddle_ball_distance_hardtrain',
        'paddle_ball_distance_hardeval'
    ]
    arg_names = [
        'env_type', 'env', 'constraint', 'reward_shaping', 'augmentation',
        'train_seed', 'eval_seed'
    ]

    args = [
        dict(zip(arg_names, arg_vals)) for arg_vals in itertools.product(
            env_type, env, constraint, reward_shaping, augmentation,
            train_seed, eval_seed)
    ]
    args = [
        d for d in args
        if not ('hard' in d['constraint'] and d['reward_shaping'] != 0)
    ]
    return [EvalTask(**a) for a in args]


def spaceinvader_tasks():
    # per-task info
    env_type = ['atari']
    env = ['SpaceInvaders']
    constraint = [
        'dangerzone', 'dangerzone_dense', 'dangerzone_hard',
        'dangerzone_hardtrain', 'dangerzone_hardeval'
    ]
    arg_names = [
        'env_type', 'env', 'constraint', 'reward_shaping', 'augmentation',
        'train_seed', 'eval_seed'
    ]

    args = [
        dict(zip(arg_names, arg_vals)) for arg_vals in itertools.product(
            env_type, env, constraint, reward_shaping, augmentation,
            train_seed, eval_seed)
    ]
    args = [
        d for d in args
        if not ('hard' in d['constraint'] and d['reward_shaping'] != 0)
    ]
    return [EvalTask(**a) for a in args]


def seaquest_tasks():
    # per-task info
    env_type = ['atari']
    env = ['Seaquest']
    constraint = ['diver', 'diver_dense', 'diver_hard']
    arg_names = [
        'env_type', 'env', 'constraint', 'reward_shaping', 'augmentation',
        'train_seed', 'eval_seed'
    ]

    args = [
        dict(zip(arg_names, arg_vals)) for arg_vals in itertools.product(
            env_type, env, constraint, reward_shaping, augmentation,
            train_seed, eval_seed)
    ]
    args = [
        d for d in args
        if not ('hard' in d['constraint'] and d['reward_shaping'] != 0)
    ]
    return [EvalTask(**a) for a in args]


def safety_gym_tasks():
    env = [
        'PointGoal1', 'PointGoal2', 'PointButton1', 'PointButton2', 'CarGoal1',
        'CarGoal2', 'CarButton1', 'CarButton2'
    ]
    constraint = ['proximity']  #'simple', 'proximity']
    seed = train_seed
    arg_names = [
        'env_type',
        'env',
        'constraint',
        'reward_shaping',
        'augmentation',
        'seed'  #'train_seed', 'eval_seed'
    ]
    args = [
        dict(zip(arg_names, arg_vals)) for arg_vals in itertools.product(
            env_type, env, constraint, reward_shaping, augmentation,
            seed)  #train_seed, eval_seed)
    ]
    args = [
        d for d in args
        if not ('hard' in d['constraint'] and d['reward_shaping'] != 0)
    ]
    return [TrainTask(**a) for a in args]


def create_tasks():
    tasks = safety_gym_tasks()  #oned_tasks() + twod_tasks() + breakout_tasks(
    #) + spaceinvader_tasks()  # + seaquest_tasks()
    random.shuffle(tasks)
    return tasks


class TrainTask(luigi.Task):
    env_type = luigi.Parameter()
    env = luigi.Parameter()
    constraint = luigi.Parameter()
    reward_shaping = luigi.IntParameter()
    augmentation = luigi.Parameter()
    seed = luigi.IntParameter()
    base_path = luigi.Parameter(significant=False)
    alternate_base_paths = luigi.ListParameter(default=[], significant=False)

    def get_task_name_str(self):
        sep = '_'
        id_str = ''
        id_str += self.env + sep
        id_str += self.constraint + sep
        id_str += str(self.reward_shaping) + sep
        id_str += self.augmentation + sep
        id_str += str(self.seed)
        return id_str

    def get_output_dir(self, base_path=None):
        if base_path is None:
            base_path = self.base_path
        return osp.join(str(base_path).strip("'"), self.get_task_name_str())

    def run(self):
        cmd_str = 'python -m baselines.run'
        if self.env_type == 'atari':
            cmd_str += ' --env ' + str(self.env) + "NoFrameskip-v4"
            cmd_str += ' --alg deepq'
            cmd_str += ' --num_timesteps 1e7'
        elif self.env_type == 'mujoco':
            # change this if necessary
            cmd_str += ' --env ' + str(self.env) + "NoFrameskip-v4"
            cmd_str += ' --alg ppo2'
            cmd_str += ' --num_timesteps 1e6'
        elif self.env_type == 'safety_gym':
            cmd_str += ' --env Safexp-' + str(self.env) + "-v0"
            cmd_str += ' --alg ppo2'
            cmd_str += ' --num_timesteps 1e6'

        if self.constraint:
            if '_hard' in self.constraint:
                cmd_str += ' --constraints ' + str(
                    self.constraint[:-5]) + '_' + str(self.env)
                cmd_str += ' --is_hard'
            elif '_dense' in self.constraint:
                cmd_str += ' --constraints ' + str(
                    self.constraint[:-6]) + '_' + str(self.env)
                cmd_str += ' --reward_shaping ' + str(self.reward_shaping)
                cmd_str += ' --is_dense'
            else:
                cmd_str += ' --constraints ' + str(
                    self.constraint) + '_' + str(self.env)
                cmd_str += ' --reward_shaping ' + str(self.reward_shaping)
        if self.augmentation:
            if self.augmentation == 'constraint_state_noembed':
                cmd_str += ' --augmentation ' + str('constraint_state')
                cmd_str += ' --embed_constraint_state False'
            else:
                cmd_str += ' --augmentation ' + str(self.augmentation)
        if self.seed:
            cmd_str += ' --seed ' + str(self.seed)
        cmd_str += ' --log_path ' + self.get_output_dir()
        cmd_str += ' --save_path ' + self.get_output_dir() + '/model'
        cmd_str += ' --num_hidden 256'
        cmd_str += ' --nsteps 32000'

        print(cmd_str)
        output = subprocess.check_output(cmd_str.split(' '))
        print(output)

    def output(self):
        return {
            "success":
            luigi.LocalTarget(osp.join(self.get_output_dir(),
                                       'final_log.txt')),
            "model":
            luigi.LocalTarget(osp.join(self.get_output_dir(), 'model'))
        }

    def complete(self):
        local_complete = luigi.LocalTarget(
            osp.join(self.get_output_dir(), 'final_log.txt')).exists()
        remote_complete = any([
            luigi.LocalTarget(
                osp.join(self.get_output_dir(base_path),
                         'final_log.txt')).exists()
            for base_path in self.alternate_base_paths
        ])
        print([
            osp.join(self.get_output_dir(base_path), 'final_log.txt')
            for base_path in self.alternate_base_paths
        ])
        return local_complete or remote_complete


if __name__ == "__main__":
    tasks = create_tasks()
    luigi.build(tasks, scheduler_url="http://localhost:8082")
    #, scheduler_url="https://pyroclast-luigi.gilgi.org/")
