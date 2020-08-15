import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas

csv_location = "./safety_gym_stats_normd.csv"
output_dir = "./perfconfplots/"
EPSILON = 0.025

df = pandas.read_csv(csv_location)

# plot scatter points
for env in ['PointGoal1', 'PointGoal2', 'PointButton1', 'PointButton2']:
    df2 = df[df['task'] == env]

    fig = plt.figure()
    fig.suptitle("{} Performance-Conformance Curve".format(env))
    plt.xlabel('Normalized Episodic Cost')
    plt.ylabel('Normalized Episodic Return')
    axes = plt.gca()

    for constraint, color in zip(['simple', 'proximity'], ['blue', 'green']):
        df3 = df2[df2['constraint'] == constraint]
        plt.scatter(df3['mean_norm_episodic_cost'],
                    df3['mean_norm_episodic_return'],
                    c=color,
                    label=constraint)

    axes.legend()

    # add 1:1 decay line
    min_val = max(axes.get_xlim()[0], axes.get_ylim()[0])
    max_val = min(axes.get_xlim()[1], axes.get_ylim()[1])
    plt.plot((min_val, max_val), (min_val, max_val),
             linestyle='dashed',
             color='grey')

    # find pareto points
    cost_return_pairs = sorted(
        list(zip(df2.mean_norm_episodic_cost, df2.mean_norm_episodic_return)))
    pareto_points = []
    best_reward = -999.
    for point in cost_return_pairs:
        if point[1] > best_reward:
            pareto_points.append(point)
            best_reward = point[1] + EPSILON

    # plot pareto points
    for i in range(len(pareto_points) - 1):
        plt.plot((pareto_points[i][0], pareto_points[i + 1][0]),
                 (pareto_points[i][1], pareto_points[i + 1][1]), 'red')

    if not Path(output_dir).exists():
        os.makedirs(output_dir)
    plt.savefig(
        str(Path(output_dir, '{}_performance_conformance.png'.format(env))))
    plt.close(fig)
