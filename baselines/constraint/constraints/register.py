from baselines.constraint.constraint import Constraint, SoftDenseConstraint
import itertools
import functools
import numpy as np

mapping = {}


def register(name):
    def _thunk(func):
        mapping[name] = func
        return func

    return _thunk


@register('1d_dithering2_Breakout')
def one_d_dithering_breakout(is_hard, reward_shaping, k=2):
    with open("./baselines/constraint/constraints/1d_dithering.lisp"
              ) as dfa_file:
        dfa_string = dfa_file.read()
    return Constraint('1d_dithering2_Breakout', dfa_string, is_hard, reward_shaping,
                      lambda obs, action, done: action)


@register('1d_dithering2_SpaceInvaders')
def one_d_dithering_spaceinvaders(is_hard, reward_shaping, k=2):
    with open("./baselines/constraint/constraints/1d_dithering.lisp"
              ) as dfa_file:
        dfa_string = dfa_file.read()
    translation_dict = dict([(0, 1), (1, 1), (2, 2), (3, 3), (4, 2), (5, 3)])
    translation_fn = lambda obs, action, done: translation_dict[action]
    return Constraint('1d_dithering2_SpaceInvaders', dfa_string, is_hard, reward_shaping,
                      translation_fn)


def build_one_d_actuation(num_actions, k):
    dfa_string_template = '(defdfa {name} (({input_symbols}) ({states}) {start_state} ({accepting_states})) ({transitions}))'
    transition_template = '({initial_state} {target_state} {symbol})'

    name = '1d_{k}_actuation'.format(k=k)
    input_symbols = ' '.join(list(map(str, range(num_actions))))
    states = ' '.join(list(map(str, range(num_actions * k +
                                          1))))  # add one for the start state
    start_state = 0
    accepting_states = ' '.join(
        [str(a * k) for a in range(1, num_actions + 1)])

    transitions = []
    for a in range(num_actions):
        transitions.append(
            transition_template.format(initial_state=0,
                                       target_state=a * k + 1,
                                       symbol=a))
        for r in range(k - 1):
            transitions.append(
                transition_template.format(initial_state=a * k + r + 1,
                                           target_state=a * k + r + 2,
                                           symbol=a))
    transitions = ' '.join(transitions)

    dfa_string = dfa_string_template.format(name=name,
                                            input_symbols=input_symbols,
                                            states=states,
                                            start_state=start_state,
                                            accepting_states=accepting_states,
                                            transitions=transitions)
    return dfa_string


@register('1d_actuation4_Breakout')
def oned_actuation_breakout4(is_hard, reward_shaping):
    return Constraint('1d_actuation_breakout4',
                      build_one_d_actuation(4, k=4),
                      is_hard,
                      reward_shaping,
                      translation_fn=lambda obs, action, done: action)


@register('1d_actuation4_SpaceInvaders')
def oned_actuation_spaceinvaders4(is_hard, reward_shaping):
    translation_dict = dict([(0, 0), (1, 1), (2, 2), (3, 3), (4, 2), (5, 3)])
    translation_fn = lambda obs, action, done: translation_dict[action]
    return Constraint('1d_actuation_SpaceInvaders',
                      build_one_d_actuation(4, k=4),
                      is_hard,
                      reward_shaping,
                      translation_fn=translation_fn)

@register("2d_actuation4_Seaquest")
def twod_actuation4_seaquest(is_hard, reward_shaping):
    with open("./baselines/constraint/constraints/seaquest_actuation.lisp"
            ) as dfa_file:
        dfa_string = dfa_file.read()
    return Constraint('2d_actuation4_Seaquest',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      translation_fn=lambda obs, action, done: action)

@register("2d_dithering4_Seaquest")
def twod_dithering4_seaquest(is_hard, reward_shaping):
    with open("./baselines/constraint/constraints/seaquest_dithering.lisp"
            ) as dfa_file:
        dfa_string = dfa_file.read()
    return Constraint('2d_dithering4_Seaquest',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      translation_fn=lambda obs, action, done: action)

@register('1d_actuation_dense')
def one_d_actuation_dense(is_hard, reward_shaping):
    ACTUATION1D_REGEX = lambda k: '2{k}|3{k}'.format(k=k)
    return CountingPotentialConstraint('1d_actuation_dense',
                                       ACTUATION1D_REGEX(4),
                                       is_hard,
                                       reward_shaping,
                                       gamma=0.99,
                                       s_active=False)


@register('sokoban_idempotent')
def sokoban_idempotent(reward_shaping):
    SOKOBAN_REGEX = '0|56|65|78|87|5678|5687|5768|5786|5867|5876|6578|6587|6758|6785|6857|6875|7568|7586|7658|7685|7856|7865|8567|8576|8657|8675|8756|8765'
    return Constraint('sokoban_idempotent',
                      SOKOBAN_REGEX,
                      reward_shaping,
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


def get_constraint(name):
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        print("Available constraints", mapping.keys())
        raise ValueError("Unknown constraint type:", name)
