import argparse
import csv
import glob
import json
import os
import os.path as osp
from pathlib import Path
import pandas

import numpy as np

EPISODE_LENGTH = 1000
NUM_EPISODES = 100


def get_params(directory):
    with open(directory / 'args.json', 'r') as args_file:
        args = json.load(args_file)
    params = dict()
    params['task'] = args['env'].split('-')[-2]
    params['seed'] = args['seed']
    params['constraint'] = ''.join(args['constraints'][0].split('_')[:-1])
    params['reward_shaping'] = args['reward_shaping'][0]
    params['is_dense'] = args['is_dense']
    params['augmentation'] = args['augmentation']
    return params


def calculate_stats(directory, num_episodes):
    cost = np.load(str(Path(directory, 'cost.npy')))
    raw_reward = np.load(str(Path(directory, 'raw_reward.npy')))
    assert abs(cost.shape[0] -
               raw_reward.shape[0]) < 1002, "cost: {}, raw_reward: {}".format(
                   cost.shape, raw_reward.shape)

    results = {}
    results['cost_rate'] = sum(cost) / cost.shape[0]
    results['episodic_cost'] = sum(
        cost[-1 * num_episodes * EPISODE_LENGTH:]) / num_episodes
    results['episodic_return'] = sum(
        raw_reward[-1 * num_episodes * EPISODE_LENGTH:]) / num_episodes
    return results


def norm_final_stats(stat_dict):
    tasks = set([name.split('_')[0] for name in stat_dict.keys()])

    #print([[k for k in stat_dict.keys() if t in k and '_0__' in k]
    #       for t in tasks])
    task_epreturn_baselines = dict([(t,
                                     np.mean([
                                         v['episodic_return']
                                         for k, v in stat_dict.items()
                                         if t in k and '_0__' in k
                                     ])) for t in tasks])
    task_epcost_baselines = dict((t,
                                  np.mean([
                                      v['episodic_cost']
                                      for k, v in stat_dict.items()
                                      if t in k and '_0__' in k
                                  ])) for t in tasks)
    task_costrate_baselines = dict((t,
                                    np.mean([
                                        v['cost_rate']
                                        for k, v in stat_dict.items()
                                        if t in k and '_0__' in k
                                    ])) for t in tasks)
    print('Return characteristic', task_epreturn_baselines)
    print('Cost characteristic', task_epcost_baselines)
    print('Cost Rate characteristic', task_costrate_baselines)
    parse_task = lambda x: x.split('_')[0]
    for k, v in stat_dict.items():
        task = parse_task(k)
        v['norm_episodic_return'] = v[
            'episodic_return'] / task_epreturn_baselines[task]
        v['norm_episodic_cost'] = v['episodic_cost'] / task_epcost_baselines[
            task]
        v['norm_cost_rate'] = v['cost_rate'] / task_costrate_baselines[task]

    return stat_dict


def agg_stats_dict(stat_dict):
    nonseed_params = set([
        '_'.join(name.split('_')[:-1]).strip('_') for name in stat_dict.keys()
    ])

    agg_stats = dict()
    for params in nonseed_params:
        agg_dict = dict()
        values_tuples = [v for k, v in stat_dict.items() if params in k]
        agg_dict['task'] = values_tuples[0]['task']
        agg_dict['constraint'] = values_tuples[0]['constraint']
        agg_dict['reward_shaping'] = values_tuples[0]['reward_shaping']
        agg_dict['is_dense'] = values_tuples[0]['is_dense']
        agg_dict['augmentation'] = values_tuples[0]['augmentation']
        agg_dict['mean_episodic_cost'] = np.mean(
            [v['episodic_cost'] for v in values_tuples])
        agg_dict['mean_norm_episodic_cost'] = np.mean(
            [v['norm_episodic_cost'] for v in values_tuples])
        agg_dict['mean_episodic_return'] = np.mean(
            [v['episodic_return'] for v in values_tuples])
        agg_dict['mean_norm_episodic_return'] = np.mean(
            [v['norm_episodic_return'] for v in values_tuples])
        agg_dict['mean_cost_rate'] = np.mean(
            [v['cost_rate'] for v in values_tuples])
        agg_dict['mean_norm_cost_rate'] = np.mean(
            [v['norm_cost_rate'] for v in values_tuples])
        agg_stats[params] = agg_dict
    return agg_stats


def augment_monitor_file(directory, viols_filepath):
    viols = np.load(viols_filepath)
    cost = np.load(str(Path(directory, 'cost.npy')))

    # copy over the first line of json
    with open(Path(directory) / '0.0.monitor.csv', 'r') as monitor:
        with open(Path(directory) / '0.0.monitor.csv.tmp',
                  'w') as temp_monitor:
            temp_monitor.writelines([monitor.readline()])

    # augment the rest of the csv
    with open(Path(directory) / '0.0.monitor.csv', 'r') as monitor:
        csvreader = csv.reader(monitor)
        with open(Path(directory) / '0.0.monitor.csv.tmp',
                  'a') as temp_monitor:
            csvwriter = csv.writer(temp_monitor)
            # skip first line
            next(csvreader)
            # add violations to the header row
            header_row = next(csvreader)
            if 'c' in header_row or 'v' in header_row:
                return
            csvwriter.writerow(header_row + ['c', 'v'])

            last_idx = 0
            for row in csvreader:
                high_idx = last_idx + int(row[1])
                row.append(str(sum(cost[last_idx:high_idx])))
                row.append(str(sum(viols[last_idx:high_idx])))
                csvwriter.writerow(row)
                last_idx = high_idx

    os.rename(
        Path(directory) / '0.0.monitor.csv',
        Path(directory) / '0.0.monitor.csv.orig')
    os.rename(
        Path(directory) / '0.0.monitor.csv.tmp',
        Path(directory) / '0.0.monitor.csv')


def main(args):
    for d in args.dirs:
        if not osp.exists(Path(d) / 'final_log.txt') or osp.exists(
                Path(d) / 'safety_gym_results.json'):
            continue

        viols_filepath = glob.glob(str(Path(d, '*_viols.npy')))[0]
        augment_monitor_file(d, viols_filepath)

        stat_dict = calculate_stats(d, NUM_EPISODES)
        with open(Path(d, 'safety_gym_results.json'), 'w') as jsonfile:
            json.dump(stat_dict, jsonfile)

    all_stats = {}
    for d in args.dirs:
        if not osp.exists(Path(d) / 'safety_gym_results.json'):
            continue
        with open(Path(d) / 'safety_gym_results.json') as result_json:
            stat_dict = json.load(result_json)
        stat_dict.update(get_params(Path(d)))
        name = d.split('/')[-1]
        all_stats[name] = stat_dict

    norm_stat_dict = norm_final_stats(all_stats)
    aggnorm_stat_dict = agg_stats_dict(norm_stat_dict)
    with open('safety_gym_stats_normd.csv', 'w') as norm_stat_csv:
        pandas.DataFrame(aggnorm_stat_dict.values()).to_csv(norm_stat_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dirs', nargs='*')
    args = parser.parse_args()
    main(args)
