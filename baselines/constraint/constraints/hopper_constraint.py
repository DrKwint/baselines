import joblib
import json
import numpy as np
import string
import functools

from baselines.constraint.constraint import CountingPotentialConstraint
from baselines.constraint.constraints.register import register, mapping


def joinit(iterable, delimiter):
    it = iter(iterable)
    for x in it:
        yield delimiter
        yield x


def mujoco_dithering(reward_shaping, joint, name, k=3):
    def idx_sign(act, idx):
        s = np.sign(act[idx])
        if s > 0:
            return '2'
        else:
            return '3'

    dithering_k = lambda k: '(23){k}|(32){k}'.format(k=k)
    s_tl = lambda o: 0
    a_tl = lambda a: idx_sign(a, joint)

    return CountingPotentialConstraint('{}_dithering_{}'.format(name, joint),
                                       dithering_k(k),
                                       reward_shaping,
                                       0.99,
                                       s_tl,
                                       a_tl,
                                       s_active=False)


mapping['hopper_dithering_0'] = functools.partial(mujoco_dithering,
                                                  joint=0,
                                                  name='hopper')
mapping['hopper_dithering_1'] = functools.partial(mujoco_dithering,
                                                  joint=1,
                                                  name='hopper')
mapping['hopper_dithering_2'] = functools.partial(mujoco_dithering,
                                                  joint=2,
                                                  name='hopper')


@register('hopper_learned')
def hopper_learned_constraint(reward):
    name = "hopper_learned"

    def hopper_state(state):
        if state[0] <= 1.:  # and state[6] <= -1.537:
            print('active')
            return 'a'
        else:
            return 'b'

    pca = joblib.load('pca.joblib')
    n_components = pca.n_components_

    def action_translation_fn(action, bucket_size=2., num_buckets=3):
        if len(action.shape) < 2:
            action = np.expand_dims(action, axis=0)
        # get everything as integers in the range [0, 2*num_buckets]
        discrete_action = np.rint(
            np.clip((pca.transform(action) / bucket_size) + num_buckets, 0,
                    2 * num_buckets)).astype(np.int)
        action_id = 0
        for i in range(n_components):
            action_id += discrete_action[:, i] * (2 * num_buckets + 1)**i
        #print(string.ascii_letters[action_id[0]])
        return list(map(lambda x: string.ascii_letters[x], action_id))

    with open('regex.txt', 'r') as regex_file:
        HOPPER_REGEX_LIST = json.load(regex_file)
    regex = '|'.join(
        list(map(lambda x: ''.join(list(joinit(x, 'a'))), HOPPER_REGEX_LIST)))
    regex = regex[:1000]
    print(regex)
    print("Hopper regex len:", len(regex))

    return CountingPotentialConstraint(name,
                                       regex,
                                       reward,
                                       0.99,
                                       s_tl=hopper_state,
                                       a_tl=action_translation_fn)
