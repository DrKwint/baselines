import os

import numpy as np

import baselines.constraint
import gym
from baselines.constraint.bench.step_monitor import LogBuffer
import collections
from gym.spaces.box import Box

#TODO: Add documentation describing what's happening here
class ConstraintEnv(gym.Wrapper):
    def __init__(self,
                 env,
                 constraints,
                 augmentation_type=None,
                 log_dir=None,
                 action_history_size=10):
        gym.Wrapper.__init__(self, env)
        if augmentation_type == 'constraint_state_concat' and isinstance(
                env.observation_space, Box):
            constraint_shape_len = sum([c.num_states for c in constraints])
            new_shape = list(env.observation_space.shape)
            new_shape[-1] = new_shape[-1] + constraint_shape_len
            self.observation_space = Box(-np.inf, np.inf, tuple(new_shape),
                                         env.observation_space.dtype)
        self.constraints = constraints
        self.augmentation_type = augmentation_type
        self.prev_obs = self.env.reset()
        self.action_history = collections.deque([0] * 10)
        if log_dir is not None:
            self.log_dir = log_dir
            self.viol_log_dict = dict([(c, LogBuffer(1024, (), dtype=np.bool))
                                       for c in constraints])
            self.state_log_dict = dict([(c, LogBuffer(1024, (), dtype=np.int32))
                                       for c in constraints])
            self.rew_mod_log_dict = dict([
                (c, LogBuffer(1024, (), dtype=np.float32)) for c in constraints
            ])
        else:
            self.logs = None

    def augment_obs(self, ob):
        """Augment the current observation.

        Parameters
        ----------
        self: ConstraintEnv
            A data structure that serves as 
            a wrapper to the Gym environment.

        ob: Tensor?
            A Tensor? or data structure that
            represents the observation of the
            state.
        
        Returns
        -------
        ob: Tensor?
            An Tensor? or data structure that 
            is augmented either via concatenation 
            or otherwise with either the constraint
            state, or the action history.
        """
        if self.augmentation_type == 'constraint_state_concat':
            ob = np.concatenate(
                [ob] + np.array([c.current_state for c in self.constraints]))
        elif self.augmentation_type == 'constraint_state_product':
            ob = (ob, np.array([c.current_state for c in self.constraints]))
        elif self.augmentation_type == 'action_history_product':
            ob = (ob, np.array(self.action_history))
        return ob

    def reset(self, **kwargs):
        """Reset the environment and update logs.

        Parameters
        ----------
        self: ConstraintEnv
            A data structure that serves as 
            a wrapper to the Gym environment.
        **kwargs: None
            A data structure that allows for
            the passing of additional arguments
            that are keyworded into the function.

        Returns
        -------
        ob: Tensor?
            An Tensor? or data structure that 
            is augmented using the augment_obs
            function after being reset.
        """    
        [c.reset() for c in self.constraints]
        [
            log.save(os.path.join(self.log_dir, c.name + '_viols'))
            for (c, log) in self.viol_log_dict.items()
        ]
        [
            log.save(os.path.join(self.log_dir, c.name + '_state'))
            for (c, log) in self.state_log_dict.items()
        ]
        [
            log.save(os.path.join(self.log_dir, c.name + '_rew_mod'))
            for (c, log) in self.rew_mod_log_dict.items()
        ]
        ob = self.env.reset(**kwargs)
        ob = self.augment_obs(ob)
        self.prev_obs = ob
        return ob

    def step(self, action):
        """
        Perform a step in the environment and 
        evaluate if any constraints have been violated.

        Parameters
        ----------
        self: ConstraintEnv
            A data structure that serves as 
            a wrapper to the Gym environment.
        act: ActWrapper
            A function that takes a batch of
            observations and returns actions.

        Returns
        -------
        ob: Tensor?
            An Tensor? or data structure that 
            is augmented using the augment_obs
            function after being reset.
        rew: 
            The reward provided by the environment.
        done: Boolean?
            Whether or not the episode is finished.
        info: ??
            Unclear.
        """
        ob, rew, done, info = self.env.step(action)
        info['raw_reward'] = rew
        if type(action) is np.ndarray:
            action = action[0]
        self.action_history.append(action)
        self.action_history.popleft()
        for c in self.constraints:
            is_vio, rew_mod = c.step(self.prev_obs, action, done)
            rew += rew_mod
            if self.viol_log_dict is not None:
                self.viol_log_dict[c].log(is_vio)
                self.state_log_dict[c].log(c.current_state)
                self.rew_mod_log_dict[c].log(rew_mod)

        ob = self.augment_obs(ob)
        self.prev_obs = ob

        return ob, rew, done, info

    def __del__(self):
        self.reset()
