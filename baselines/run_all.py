import luigi
from baselines.slurm_task import SlurmTask, SlurmInfo
import itertools

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


class ProjectSlurmTask(SlurmTask):
    def ex_hpc(self, command):
        command = ['module', 'load', 'anaconda;'
                   ] + ['conda', 'activate', 'tf1.14'] + command
        return super(ProjectSlurmTask, self).ex_hpc(command)


class TrainModelTask(ProjectSlurmTask):
    env = luigi.Parameter()
    alg = luigi.Parameter()
    num_timesteps = luigi.IntParameter()
    constraints = luigi.ListParameter()
    reward_shaping = luigi.ListParameter()
    seed = luigi.IntParameter()

    def requires(self):
        pass

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
        pass


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
