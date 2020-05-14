import functools
import itertools

import numpy as np
from skimage.feature import match_template
import skimage
import math

from baselines.common.atari_wrappers import LazyFrames
from baselines.constraint.constraint import Constraint, SoftDenseConstraint

mapping = {}


def register(name):
    def _thunk(func):
        mapping[name] = func
        return func

    return _thunk


@register('dodge_bullet_SpaceInvaders')
def dodge_bullet_spaceinvaders(is_hard, is_dense, reward_shaping):
    with open("./baselines/constraint/constraints/1d_dithering.lisp"
              ) as dfa_file:
        dfa_string = dfa_file.read()

    bullet_detection_baseline = None
    player_detection_baseline = None
    player_template = np.array([[0, 0, 6, 79, 28, 0, 0],
                                [0, 0, 30, 98, 67, 0, 0],
                                [0, 6, 55, 98, 77, 21, 0],
                                [0, 12, 91, 98, 98, 41, 0]])
    bullet_template = np.array([[0, 40, 0], [0, 99, 0], [0, 99, 0], [0, 40,
                                                                     0]])
    translation_dict = dict([(0, 'a'), (1, 'a'), (2, 'r'), (3, 'l'), (4, 'r'),
                             (5, 'l')])
    inv_translation_dict = {'a': [0, 1], 'r': [2, 4], 'l': [3, 5]}

    def token2int(t):
        act = t[0]
        if act == 'r':
            act = 2
        elif act == 'l':
            act = 1
        elif act == 'a':
            act = 0
        l = t[2]
        r = t[4]
        a = t[6]
        return 64 * act * 16 * int(r) + 4 * int(l) + int(a)

    def translation_fn(obs, action, done):
        if not isinstance(obs, LazyFrames):
            obs = obs[0]
        frames = np.array(obs)
        nonlocal bullet_detection_baseline
        nonlocal player_detection_baseline
        old_bullet_player_area = frames[45:-5, :, -2]
        bullet_player_area = frames[45:-5, :, -1]

        old_bullet_conv = match_template(old_bullet_player_area,
                                         bullet_template,
                                         pad_input=True)
        bullet_conv = match_template(bullet_player_area,
                                     bullet_template,
                                     pad_input=True)
        player_conv = match_template(bullet_player_area,
                                     player_template,
                                     pad_input=True)
        if bullet_detection_baseline is None:
            bullet_detection_baseline = np.max(bullet_conv) + 0.02
            #print("bullet threshold", bullet_detection_baseline)
        if player_detection_baseline is None:
            player_detection_baseline = np.minimum(
                np.max(player_conv) + 0.02, 0.95)
            #print("player threshold", player_detection_baseline)

        old_bullet_locs = np.argwhere(
            old_bullet_conv > bullet_detection_baseline)
        bullet_locs = np.argwhere(bullet_conv > bullet_detection_baseline)
        player_locs = np.argwhere(player_conv > player_detection_baseline)

        if done:
            bullet_detection_baseline = None
            player_detection_baseline = None

        if len(player_locs) != 1 or len(bullet_locs) == 0:
            # player is missing or we don't where they are
            # OR no bullets detected
            # then the constraint is inactive
            return 0

        # merge individual points which belong together
        def merge_bullet_areas(bullet_locs):
            bullet_areas = [(a, a, b) for a, b in bullet_locs]
            i = 0
            while i < len(bullet_areas) - 1:
                j = i + 1
                while j < len(bullet_areas):
                    a, b, c = bullet_areas[i]
                    d, e, f = bullet_areas[j]
                    if c == f and (b == d - 1):
                        bullet_areas[i] = (a, e, c)
                        del bullet_areas[j]
                    else:
                        j += 1
                i += 1
            return bullet_areas

        bullet_areas = merge_bullet_areas(bullet_locs)
        old_bullet_areas = merge_bullet_areas(old_bullet_locs)

        # filter out bullets travelling upwards
        down_bullets = []
        for bullet in bullet_areas:
            a, b, c = bullet
            for d, e, f in old_bullet_areas:
                if c == f and d < a and e < b:
                    down_bullets.append([b, c])
                    break
        if len(down_bullets) == 0:
            return 0

        player_loc = player_locs[0]
        player_x_line = (player_loc[1] - 3, player_loc[1] + 3)
        # sort down bullets into lra
        left = []
        right = []
        above = []
        for bullet in down_bullets:
            if bullet[1] < player_x_line[0]:
                left.append(bullet)
            elif bullet[1] > player_x_line[1]:
                right.append(bullet)
            else:
                above.append(bullet)

        distance = lambda p, b: (b[0] - p[0])**2 + (b[1] - p[1])**2
        token_str = translation_dict[action]

        if len(left) == 0:
            token_str += 'l3'
        else:
            min_lbullet_distance = min([distance(player_loc, x) for x in left])
            if min_lbullet_distance < 12:
                token_str += 'l0'
            elif min_lbullet_distance < 24:
                token_str += 'l1'
            else:
                token_str += 'l2'
        if len(right) == 0:
            token_str += 'r3'
        else:
            min_rbullet_distance = min(
                [distance(player_loc, x) for x in right])
            if min_rbullet_distance < 12:
                token_str += 'r0'
            elif min_rbullet_distance < 24:
                token_str += 'r1'
            else:
                token_str += 'r2'
        if len(above) == 0:
            token_str += 'a3'
        else:
            min_abullet_distance = min(
                [distance(player_loc, x) for x in above])
            if min_abullet_distance < 12:
                token_str += 'a0'
            elif min_abullet_distance < 24:
                token_str += 'a1'
            else:
                token_str += 'a2'
            return token2int(token_str)

    def inv_translation_fn(token):
        i = token // 64
        if i == 2:
            c = 'r'
        elif i == 1:
            c = 'l'
        elif i == 0:
            c = 'a'
        return inv_translation_dict[c]

    if is_dense:
        return SoftDenseConstraint('dodge_bullet_dense_SpaceInvaders',
                                   dfa_string,
                                   reward_shaping,
                                   translation_fn,
                                   gamma=.99)
    return Constraint('dodge_bullet_SpaceInvaders',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      translation_fn,
                      inv_translation_fn=inv_translation_fn)


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
                                   translation_dict,
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
