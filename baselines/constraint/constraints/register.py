from baselines.constraint.constraint import Constraint, CountingPotentialConstraint
from baselines.constraint.constraints.reacher_dynamic_constraint import reacher_dynamic_constraint
import itertools
import functools
import numpy as np

mapping = {}


def register(name):
    def _thunk(func):
        mapping[name] = func
        return func

    return _thunk

@register('empty')
def empty(_):
    return Constraint('empty', '', 0, s_active=False, a_active=False)

@register('1d_dithering')
def one_d_dithering(reward_shaping, k=2):
    DITHERING1D_REGEX = lambda k: '(23){k}|(32){k}'.format(k=k)
    return Constraint('1d_dithering',
                      DITHERING1D_REGEX(k),
                      reward_shaping,
                      s_active=False)


@register('1d_dithering_dense')
def one_d_dithering_dense(reward_shaping, k=2):
    DITHERING1D_REGEX = lambda k: '(23){k}|(32){k}'.format(k=k)
    return CountingPotentialConstraint('1d_dithering_dense',
                                       DITHERING1D_REGEX(k),
                                       reward_shaping,
                                       gamma=0.99,
                                       s_active=False)


@register('1d_actuation')
def one_d_actuation(reward_shaping):
    ACTUATION1D_REGEX = lambda k: '2{k}|3{k}'.format(k=k)
    return Constraint('1d_actuation',
                      ACTUATION1D_REGEX(4),
                      reward_shaping,
                      s_active=False)


@register('1d_actuation_dense')
def one_d_actuation_dense(reward_shaping):
    ACTUATION1D_REGEX = lambda k: '2{k}|3{k}'.format(k=k)
    return CountingPotentialConstraint('1d_actuation_dense',
                                       ACTUATION1D_REGEX(4),
                                       reward_shaping,
                                       gamma=0.99,
                                       s_active=False)


# see below for registration
def half_cheetah_dithering(reward_shaping, joint, k=3):
    def idx_sign(act, idx):
        s = np.sign(act[idx])
        if s > 0:
            return '2'
        else:
            return '3'

    half_cheetah_dithering_k = lambda k: '(23){k}|(32){k}'.format(k=k)
    s_tl = lambda o: 0
    a_tl = lambda a: idx_sign(a, joint)

    return CountingPotentialConstraint(
        'half_cheetah_dithering_{}'.format(joint),
        half_cheetah_dithering_k(k),
        reward_shaping,
        0.99,
        s_tl,
        a_tl,
        s_active=False)

mapping['half_cheetah_dithering_0'] = functools.partial(half_cheetah_dithering,
                                                        joint=0)
mapping['half_cheetah_dithering_1'] = functools.partial(half_cheetah_dithering,
                                                        joint=1)
mapping['half_cheetah_dithering_2'] = functools.partial(half_cheetah_dithering,
                                                        joint=2)
mapping['half_cheetah_dithering_3'] = functools.partial(half_cheetah_dithering,
                                                        joint=3)
mapping['half_cheetah_dithering_4'] = functools.partial(half_cheetah_dithering,
                                                        joint=4)
mapping['half_cheetah_dithering_5'] = functools.partial(half_cheetah_dithering,
                                                        joint=5)


mapping['half_cheetah_dithering_0'] = functools.partial(half_cheetah_dithering,
                                                        joint=0)
mapping['half_cheetah_dithering_0'] = functools.partial(half_cheetah_dithering,
                                                        joint=1)
mapping['half_cheetah_dithering_0'] = functools.partial(half_cheetah_dithering,
                                                        joint=2)
mapping['half_cheetah_dithering_0'] = functools.partial(half_cheetah_dithering,
                                                        joint=3)
mapping['half_cheetah_dithering_0'] = functools.partial(half_cheetah_dithering,
                                                        joint=4)
mapping['half_cheetah_dithering_0'] = functools.partial(half_cheetah_dithering,
                                                        joint=5)


def get_constraint(name):
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        print("Available constraints", mapping.keys())
        raise ValueError("Unknown constraint type:", name)
