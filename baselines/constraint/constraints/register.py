from baselines.constraint.constraint import Constraint, CountingPotentialConstraint
from baselines.constraint.constraints.reacher_dynamic_constraint import reacher_dynamic_constraint

mapping = {}

def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk

@register('1d_dithering')
def one_d_dithering(reward_shaping, k=2):
    DITHERING1D_REGEX = lambda k: '(23){k}|(32){k}'.format(k=k)
    return Constraint('1d_dithering', DITHERING1D_REGEX(k), reward_shaping, s_active=False)

@register('1d_dithering_dense')
def one_d_dithering_dense(reward_shaping, k=2):
    DITHERING1D_REGEX = lambda k: '(23){k}|(32){k}'.format(k=k)
    return CountingPotentialConstraint('1d_dithering_dense', DITHERING1D_REGEX(k), reward_shaping, gamma=0.99, s_active=False)

@register('1d_actuation')
def one_d_actuation(reward_shaping):
    ACTUATION1D_REGEX = lambda k: '2{k}|3{k}'.format(k=k)
    return Constraint('1d_actuation', ACTUATION1D_REGEX(4), reward_shaping, s_active=False)

@register('1d_actuation_dense')
def one_d_actuation_dense(reward_shaping):
    ACTUATION1D_REGEX = lambda k: '2{k}|3{k}'.format(k=k)
    return CountingPotentialConstraint('1d_actuation_dense', ACTUATION1D_REGEX(4), reward_shaping, gamma=0.99, s_active=False)

def get_constraint(name):
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        print("Available constraints", mapping.keys())
        raise ValueError("Unknown constraint type:", name)
