import functools
import itertools

import numpy as np

from baselines.common.atari_wrappers import LazyFrames
from baselines.constraint.constraint import Constraint, SoftDenseConstraint

mapping = {}


def register(name):
    def _thunk(func):
        mapping[name] = func
        return func

    return _thunk


@register('paddle_ball_distance_Breakout')
def paddle_direction_breakout(is_hard, is_dense, reward_shaping):
    with open(
            "./baselines/constraint/constraints/int_counter_with_null_reset.lisp"
    ) as dfa_file:
        dfa_string = dfa_file.read()

    limit = 10
    trigger_pulled = False

    def translation_fn(obs, action, done):
        # action 0 is noop
        # action 1 is the fire button
        # action 2 goes to the right
        # action 3 goes to the left
        if not isinstance(obs, LazyFrames):
            obs = obs[0]
        frames = np.array(obs)
        paddle_line = frames[-7, 5:-5, -1]
        ball_box = frames[38:-9, 5:-5, -1]
        ball_pixels = np.nonzero(ball_box)
        paddle_pixels = np.nonzero(paddle_line)
        nonlocal trigger_pulled
        if action == 1:
            trigger_pulled = True
        if done:
            trigger_pulled = False
        if not trigger_pulled:
            return 'N'
        try:
            ball_x_center = (np.min(ball_pixels[1]) +
                             np.max(ball_pixels[1])) / 2.
            paddle_x_center = (np.min(paddle_pixels[0]) +
                               np.max(paddle_pixels[0])) / 2.
            # case where paddle is too far to the right
            if ball_x_center - paddle_x_center < -limit and (action != 3):
                return 1
            # too far to the left
            elif ball_x_center - paddle_x_center > limit and (action != 2):
                return -1
            else:
                return 0
        except ValueError:
            return 'N'

    def inv_translation_fn(token):
        if token == 1:
            return [0, 1, 2]
        elif token == -1:
            return [0, 1, 3]
        else:
            print(token)
            exit()

    if is_dense:
        return SoftDenseConstraint('paddle_direction_dense_Breakout',
                                   dfa_string,
                                   reward_shaping,
                                   translation_fn,
                                   gamma=0.99)
    return Constraint('paddle_direction_Breakout',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      translation_fn=translation_fn,
                      inv_translation_fn=inv_translation_fn)


@register('1d_dithering2_Breakout')
def one_d_dithering_breakout(is_hard, is_dense, reward_shaping, k=2):
    with open("./baselines/constraint/constraints/1d_dithering.lisp"
              ) as dfa_file:
        dfa_string = dfa_file.read()
    if is_dense:
        return SoftDenseConstraint('1d_dithering2_dense_Breakout',
                                   dfa_string,
                                   reward_shaping,
                                   lambda obs, action, done: action,
                                   gamma=0.99)
    return Constraint('1d_dithering2_Breakout',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      lambda obs, action, done: action,
                      inv_translation_fn=lambda token: [token])


@register('1d_dithering2_SpaceInvaders')
def one_d_dithering_spaceinvaders(is_hard, is_dense, reward_shaping, k=2):
    with open("./baselines/constraint/constraints/1d_dithering.lisp"
              ) as dfa_file:
        dfa_string = dfa_file.read()
    translation_dict = dict([(0, 1), (1, 1), (2, 2), (3, 3), (4, 2), (5, 3)])
    inv_translation_dict = {1: [0, 1], 2: [2, 4], 3: [3, 5]}
    translation_fn = lambda obs, action, done: translation_dict[action]
    inv_translation_fn = lambda token: inv_translation_dict[token]
    if is_dense:
        return SoftDenseConstraint('1d_dithering2_dense_Breakout',
                                   dfa_string,
                                   reward_shaping,
                                   lambda obs, action, done: action,
                                   gamma=.99)
    return Constraint('1d_dithering2_SpaceInvaders',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      translation_fn,
                      inv_translation_fn=inv_translation_fn)


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
def oned_actuation_breakout4(is_hard, is_dense, reward_shaping):
    if is_dense:
        return SoftDenseConstraint(
            '1d_actuation_dense_breakout4',
            build_one_d_actuation(4, k=4),
            reward_shaping,
            translation_fn=lambda obs, action, done: action,
            gamma=0.99)
    return Constraint('1d_actuation_breakout4',
                      build_one_d_actuation(4, k=4),
                      is_hard,
                      reward_shaping,
                      translation_fn=lambda obs, action, done: action,
                      inv_translation_fn=lambda token: [token])


@register('1d_actuation4_SpaceInvaders')
def oned_actuation_spaceinvaders4(is_hard, is_dense, reward_shaping):
    translation_dict = dict([(0, 0), (1, 1), (2, 2), (3, 3), (4, 2), (5, 3)])
    inv_translation_dict = {0: [0], 1: [1], 2: [2, 4], 3: [3, 5]}
    translation_fn = lambda obs, action, done: translation_dict[action]
    inv_translation_fn = lambda token: inv_translation_dict[token]
    if is_dense:
        return SoftDenseConstraint('1d_actuation_dense_SpaceInvaders',
                                   build_one_d_actuation(4, k=4),
                                   reward_shaping,
                                   translation_fn=translation_fn,
                                   gamma=0.99)
    return Constraint('1d_actuation_SpaceInvaders',
                      build_one_d_actuation(4, k=4),
                      is_hard,
                      reward_shaping,
                      translation_fn=translation_fn,
                      inv_translation_fn=inv_translation_fn)


@register("2d_actuation4_Seaquest")
def twod_actuation4_seaquest(is_hard, is_dense, reward_shaping):
    with open("./baselines/constraint/constraints/seaquest_actuation.lisp"
              ) as dfa_file:
        dfa_string = dfa_file.read()
    if is_dense:
        return SoftDenseConstraint(
            '2d_actuation4_dense_Seaquest',
            dfa_string,
            reward_shaping,
            translation_fn=lambda obs, action, done: action,
            gamma=0.99)
    return Constraint('2d_actuation4_Seaquest',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      translation_fn=lambda obs, action, done: action,
                      inv_translation_fn=lambda token: [token])


@register("2d_dithering4_Seaquest")
def twod_dithering4_seaquest(is_hard, is_dense, reward_shaping):
    with open("./baselines/constraint/constraints/seaquest_dithering.lisp"
              ) as dfa_file:
        dfa_string = dfa_file.read()
    if is_dense:
        return SoftDenseConstraint(
            '2d_dithering4_dense_Seaquest',
            dfa_string,
            reward_shaping,
            translation_fn=lambda obs, action, done: action,
            gamma=0.99)
    return Constraint('2d_dithering4_Seaquest',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      translation_fn=lambda obs, action, done: action,
                      inv_translation_fn=lambda token: [token])


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
