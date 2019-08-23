import luigi
#from .slurm_task import SlurmTask, SlurmInfo
import itertools
import subprocess

cmd_line_params = {
    'env': [
        "BreakoutNoFrameskip-v4", "SpaceInvadersNoFrameskip-v4",
        "SeaquestNoFrameskip-v4"
    ],
    'alg': ['deepq'],
    'num_timesteps': ['1e7'],
    'constraints': [['1d_actuation'], ['1d_dithering'], ['1d_actuation_dense'],
                    ['1d_dithering_dense']],
    'reward_shaping': [[0], [-1], [-10], [-100]],
    'seed': [149876]
}


class SlurmTask(luigi.Task):
    command = luigi.ListParameter()
    time = luigi.Parameter()
    mem = luigi.Parameter()
    job_name = luigi.Parameter()
    partition = luigi.Parameter()
    gres = luigi.Parameter()
    log_dir = luigi.Parameter()

    def run(self):
        file_string = "#!/bin/sh\n"
        file_string += "#SBATCH --time={}\n".format(self.time)
        file_string += "#SBATCH --mem={}\n".format(self.mem)
        file_string += "#SBATCH --job-name={}\n".format(self.job_name)
        file_string += "#SBATCH --partition={}\n".format(self.partition)
        file_string += "#SBATCH --gres={}\n".format(self.gres)
        file_string += '\n'
        file_string += 'module load anaconda\n'
        file_string += 'conda activate tf1.14\n'
        file_string += 'OPENAI_LOGDIR="./openai_log/{}" '.format(self.log_dir)
        print(file_string)

    def output(self):
        return [
            luigi.LocalTarget('./openai_log/{}/log.txt'.format(self.log_dir))
        ]


class TrainModelTask(luigi.Task):
    env = luigi.Parameter()
    alg = luigi.Parameter()
    num_timesteps = luigi.IntParameter()
    constraints = luigi.ListParameter()
    reward_shaping = luigi.ListParameter()
    seed = luigi.IntParameter()

    def requires(self):
        command = ['python', '-u', '-m', 'baselines.run']
        # env
        command += ['--env', self.env]
        # alg
        command += ['--alg', self.alg]
        # num_timesteps
        command += ['--num_timesteps', str(self.num_timesteps)]
        # constraints and reward shaping
        if len(self.constraints) > 0:
            command += ['--constraints', str(self.constraints)[1:-2]]
            command += ['--reward_shaping', str(self.constraints)[1:-2]]
        # seed
        command += ['--seed', str(self.seed)]

        log_dir = '_'.join(
            [self.env, self.alg, str(self.num_timesteps)] + list(
                itertools.chain.from_iterable(
                    zip(self.constraints, map(str, self.reward_shaping)))))
        return SlurmTask(command=command,
                         time='2-00:00',
                         mem='16000',
                         job_name='rl_constraint',
                         partition='scott',
                         gres='gpu:1',
                         log_dir=log_dir)

    def run(self):
        pass

    def output(self):
        return self.input()


if __name__ == '__main__':
    all_args = [
        dict(zip(cmd_line_params.keys(), values))
        for values in itertools.product(*cmd_line_params.values())
    ]
    print(all_args[0])
    luigi.build([TrainModelTask(**all_args[0])], local_scheduler=True)
