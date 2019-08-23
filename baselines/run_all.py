import luigi
from .slurm import SlurmTask, SlurmInfo
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


class MySlurmTask(luigi.Task):
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

        with open('./slurm_job_file', 'w') as slurm_job_file:
            slurm_job_file.write(file_string)

        subprocess.run(['sbatch', './slurm_job_file'])

    def output(self):
        return [
            luigi.LocalTarget('./openai_log/{}/log.txt'.format(self.log_dir))
        ]


class TrainModelTask(SlurmTask):
    env = luigi.Parameter()
    alg = luigi.Parameter()
    num_timesteps = luigi.IntParameter()
    constraints = luigi.ListParameter()
    reward_shaping = luigi.ListParameter()
    seed = luigi.IntParameter()

    def requires(self):
        '''
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
        '''
        return

    def run(self):
        command = [
            'OPENAI_LOGDIR="./openai_log/${SLURM_JOBID}"', 'python', '-u',
            '-m', 'baselines.run'
        ]
        # env
        command += ['--env', self.env]
        # alg
        command += ['--alg', self.alg]
        # num_timesteps
        command += ['--num_timesteps', str(self.num_timesteps)]
        # constraints and reward shaping
        if len(self.constraints) > 0:
            command += ['--constraints', str(self.constraints)[1:-1]]
            command += ['--reward_shaping', str(self.constraints)[1:-1]]
        # seed
        command += ['--seed', str(self.seed)]
        self.ex_hpc(command)

    def output(self):
        return


if __name__ == '__main__':
    all_args = [
        dict(zip(cmd_line_params.keys(), values))
        for values in itertools.product(*cmd_line_params.values())
    ]
    print(all_args[0])
    slurminfo = SlurmInfo('runmode_hpc', 'constrained_rl', 'scott', '1',
                          '2-00:00', 'constrained_rl', '1 -gres=gpu:1')
    luigi.build([TrainModelTask(**all_args[0], slurminfo=slurminfo)],
                local_scheduler=True)
