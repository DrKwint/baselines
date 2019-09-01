import itertools
import subprocess

cmd_line_params = {
    'env': [
        "BreakoutNoFrameskip-v4", "SpaceInvadersNoFrameskip-v4",
        "SeaquestNoFrameskip-v4"
    ],
    'alg': ['deepq'],
    'num_timesteps': ['1e7'],
    'constraints': [
        '1d_actuation', '1d_dithering', '1d_actuation_dense',
        '1d_dithering_dense'
    ],
    'reward_shaping': [0, -1, -10, -100],
    'augmentation': ['constraint_state', 'action_history', None],
    'seed': [149876]
}

slurm_dict = {
    'time': '2-00:00:00',
    'mem': '8000',
    'job-name': 'constraint_mdp',
    'partition': 'scott',
    'gres': 'gpu:1',
    'setup_commands': ['module load anaconda', 'conda activate tf1.14'],
    'env_vars': ['OPENAI_LOGDIR="./openai_log/${SLURM_JOBID}"']
}


def assemble_cmd_string(param_dict):
    command = ['python', '-u', '-m', 'baselines.run']
    for key in param_dict.keys():
        command += ['--{}'.format(key), str(param_dict[key])]
    return command


def assemble_sbatch_file(slurm_dict, param_dict):
    file_str = '#!/bin/sh\n'
    file_str += '#SBATCH --time={}\n'.format(slurm_dict['time'])
    file_str += '#SBATCH --mem={}\n'.format(slurm_dict['mem'])
    file_str += '#SBATCH --job-name={}\n'.format(slurm_dict['job-name'])
    file_str += '#SBATCH --partition={}\n'.format(slurm_dict['partition'])
    file_str += '#SBATCH --gres={}\n\n'.format(slurm_dict['gres'])
    for line in slurm_dict['setup_commands']:
        file_str += line + '\n'

    command_list = assemble_cmd_string(param_dict)
    command_str = ' '.join(command_list)
    for env_var in slurm_dict['env_vars']:
        command_str = env_var + ' ' + command_str
    print(command_str)
    file_str += command_str
    return file_str


def run(slurm_dict, param_dict, filename='sbatch_command_file.txt'):
    sbatch_file_str = assemble_sbatch_file(slurm_dict, param_dict)
    with open(filename, 'w') as sbatch_file:
        sbatch_file.write(sbatch_file_str)
    completed_process = subprocess.run(['sbatch', filename],
                                       capture_output=True)
    jobid = int(str(completed_process.stdout).split(' ')[-1][:-3])
    return jobid


def run_all(slurm_dict, all_param_dict):
    all_args = [
        dict(zip(all_param_dict.keys(), values))
        for values in itertools.product(*all_param_dict.values())
    ]
    record_dict = {}
    for args_dict in all_args:
        job_id = run(slurm_dict, args_dict)
        record_dict[str(args_dict)] = job_id
    return record_dict


if __name__ == '__main__':
    run_all(slurm_dict, cmd_line_params)
