from baselines.constraint.constraint import CountingPotentialConstraint
import numpy as np

def reacher_dynamic_constraint(reward):
    name = "reacher_dynamic"

    def reacher_distance_to_target(obs):
        distance = np.linalg.norm(obs[-3:-1])
        discrete = int(distance // 0.1)
        if discrete < 0: discrete = 0
        if discrete > 4: discrete = 4
        return str(discrete)

    def reacher_discretize_action(act):
        norm = np.linalg.norm(act)
        discrete = int(norm // 0.65)
        if discrete < 0: discrete = 0
        if discrete > 4: discrete = 4
        return str(discrete)

    regex = "|".join(['s' + s for s in ["34","23","24","12","13","14","01","02","03","04"]])
    print(regex)

    return CountingPotentialConstraint(
        name, regex, reward, 0.99, s_tl=reacher_distance_to_target, a_tl=reacher_discretize_action)
