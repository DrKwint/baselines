import itertools
from collections import Counter

import numpy as np
import tensorflow as tf

from baselines.constraint.dfa import DFA


class Constraint(object):
    def __init__(self,
                 name,
                 dfa_string,
                 is_hard,
                 violation_reward=None,
                 translation_fn=lambda x: x):
        self.name = name
        self.dfa = DFA.from_string(dfa_string)
        if not is_hard:
            assert violation_reward is not None
        self.violation_reward = violation_reward
        self.translation_fn = translation_fn
        self.is_hard = is_hard

    def step(self, obs, action, done):
        is_viol = self.dfa.step(self.translation_fn(obs, action, done))
        rew_mod = self.violation_reward if is_viol else 0.
        return is_viol, rew_mod

    def reset(self):
        self.dfa.reset()

    @property
    def current_state(self):
        return self.dfa.current_state

    @property
    def num_states(self):
        return len(self.dfa.states)

    def is_violating(self, obs, action, done):
        return self.dfa.step(self.translation_fn(obs, action, done), hypothetical=True)

class SoftDenseConstraint(Constraint):
    def __init__(self, name, dfa_string, violation_reward, translation_fn, gamma):
        super(SoftDenseConstraint, self).__init__(name, dfa_string, False, violation_reward, translation_fn)
        self.gamma = gamma
        # counters for tracking value of each DFA state
        self.prev_state = self.current_state
        self.episode_visit_count = Counter()
        self.visit_count = Counter(self.dfa.states)
        self.violation_count = Counter(self.dfa.accepting_states)

    def get_state_potentials(self):
        potential = lambda s: self.violation_count[s] / self.visit_count[s]
        return {s: potential(s) for s in self.dfa.states}

    def step(self, obs, action, done):
        is_viol, _ = super().step(obs, action, done)
        self.episode_visit_count[self.current_state] += 1

        current_viol_propn = (self.violation_count[self.current_state] /
                              self.visit_count[self.current_state])
        prev_viol_propn = (self.violation_count[self.prev_state] /
                           self.visit_count[self.prev_state])
        rew_mod = (self.gamma * current_viol_propn -
                   prev_viol_propn) * self.violation_reward
        if self.prev_state in self.dfa.accepting_states: rew_mod = 0

        if is_viol:
            self.violation_count += self.episode_visit_count
            self.visit_count += self.episode_visit_count
            self.episode_visit_count = Counter()
        if done:
            self.visit_count += self.episode_visit_count
            self.episode_visit_count = Counter()

        self.prev_state = self.current_state
        return is_viol, rew_mod

    def reset(self):
        self.dfa.reset()
        self.prev_state = self.current_state
