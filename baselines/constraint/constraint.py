import itertools
from collections import Counter

import numpy as np
import tensorflow as tf

from baselines.constraint.dfa import DFA


class Constraint(DFA):
    """
        Constraint represents a given constraint to place on the agent.
        
        Attributes
        ==========
        name : str
            The name of the constraint
        reg_ex : str
            A representation of the constraint
            using regular expressions.
        violation_reward : int
            The reward signal for violating the
            constraint. Used in reward shaping.
        s_tl : Object, Callable(state)->str
            Stands for the state translation layer
            Takes in the state returned by the
            environment and returns
            alphabet of the DFA.
        a_tl : Object, Callable(state)->str
            Stands for the action translation layer.
            Takes in the action returned by the
            environment and returns a token in
            alphabet of the DFA.
        s_active : boolean
            A flag indicating if the state 
            translation layer is in effect.
        a_active : boolean
            A flag indicating if the action
            translation laer is in effect.

        Methods
        =======
        __init__(self, name, reg_ex,
                violation_reward, s_tl,
                a_tl, s_active, a_active)
                Initializes a Constraint.
        step(self, obs,
            action, done)
            Runs one step of the constraint mechanism
            which takes steps in paralell with the MDP
            it is attached to.
    """
    def __init__(self,
                 name,
                 reg_ex,
                 violation_reward,
                 s_tl=id,
                 a_tl=id,
                 s_active=True,
                 a_active=True):
        """
            Initializes a Constraint.

            Parameters
            ==========
            self : Constraint
                The constraint to be initalized
            name : str
                The name of the constraint
            reg_ex : str
                A representation of the constraint
                using regular expressions.
            violation_reward : int
                The reward signal for violating the
                constraint. Used in reward shaping.
            s_tl : Object, Callable(state)->str
                Stands for the state translation layer
                Takes in the state returned by the
                environment and returns
                alphabet of the DFA. 
            a_tl : Object, Callable(state)->str
                Stands for the action translation layer. 
                Takes in the action returned by the
                environment and returns a token in
                alphabet of the DFA.
            s_active : boolean
                A flag indicating if the state 
                translation layer is in effect.
            a_active : boolean
                A flag indicating if the action
                translation layer is in effect.
        """
        #TODO: Requesting clarification on the super() call.
        super(Constraint, self).__init__(reg_ex)
        self.name = name
        self.violation_reward = violation_reward
        self.s_tl = s_tl
        self.a_tl = a_tl
        self.s_active = s_active
        self.a_active = a_active

    def step(self, obs, action, done):
        """
            Take a step in the constraint mechanism.

            Runs one step of the constraint mechanism
            which takes steps in paralell with the MDP
            it is attached to.
        
            Parameters
            ----------
            #TODO: Requesting explanation of paramters
            here. Need clarification on if obs, action
            and done are the same as the one in DQN.
            
            Returns
            -------
            is_viol : boolean
                A flag indicating if the constraint is violated.
            rew_mod : float
                The modification to the reward signal
                depending on if the constraint is violated
                or not. If no violations occurred, the
                modification should be 0.
        """
        is_viol = False
        if self.s_active and self.a_active:
            is_viol = is_viol | super().step('s')
        if self.s_active:
            is_viol = is_viol | super().step(self.s_tl(obs))
        if self.a_active:
            is_viol = is_viol | super().step(self.a_tl(action))
        rew_mod = self.violation_reward if is_viol else 0.
        return is_viol, rew_mod

class CountingPotentialConstraint(Constraint):
    def __init__(self,
                 name,
                 reg_ex,
                 violation_reward,
                 gamma,
                 s_tl=id,
                 a_tl=id,
                 s_active=True,
                 a_active=True):
        super(CountingPotentialConstraint,
              self).__init__(name, reg_ex, violation_reward, s_tl, a_tl,
                             s_active, a_active)
        self.episode_visit_count = Counter()
        self.visit_count = Counter(self.states())
        self.violation_count = Counter(self.accepting_states())
        self.gamma = gamma
        self.prev_state = self.current_state

    def get_state_potentials(self):
        potential = lambda s: self.violation_count[s] / self.visit_count[s]
        return {s: potential(s) for s in self.states()}

    def step(self, obs, action, done):
        is_viol, _ = super().step(obs, action, done)
        dfa_state = self.current_state
        self.episode_visit_count[dfa_state] += 1

        current_viol_propn = (self.violation_count[dfa_state] /
                              self.visit_count[dfa_state])
        prev_viol_propn = (self.violation_count[self.prev_state] /
                           self.visit_count[self.prev_state])
        rew_mod = (self.gamma * current_viol_propn -
                   prev_viol_propn) * self.violation_reward
        if self.prev_state in self.accepting_states(): rew_mod = 0

        if is_viol:
            self.violation_count += self.episode_visit_count
            self.visit_count += self.episode_visit_count
            self.episode_visit_count = Counter()
        if done:
            self.visit_count += self.episode_visit_count
            self.episode_visit_count = Counter()

        self.prev_state = dfa_state
        return is_viol, rew_mod
