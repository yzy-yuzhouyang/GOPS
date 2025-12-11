from collections import defaultdict, deque

import gymnasium as gym
import numpy as np
import torch

from dm_control import suite
suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks('custom')
suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)
from dm_control.suite.wrappers import action_scale

class Timeout(gym.Wrapper):
    """
    Wrapper for enforcing a time limit on the environment.
    """

    def __init__(self, env, max_episode_steps):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
    
    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    def reset(self, **kwargs):
        self._t = 0
        info = {}
        return self.env.reset(**kwargs), info

    def step(self, action):
        obs, reward, te, tr, info = self.env.step(action)
        self._t += 1
        tr = self._t >= self.max_episode_steps
        return obs, reward, te, tr, info


def get_obs_shape(env):
    obs_shp = []
    for v in env.observation_spec().values():
        try:
            shp = np.prod(v.shape)
        except:
            shp = 1
        obs_shp.append(shp)
    return (int(np.sum(obs_shp)),)


class DMControlWrapper:
    def __init__(self, env, domain):
        self.env = env
        self.camera_id = 2 if domain == 'quadruped' else 0
        obs_shape = get_obs_shape(env)
        action_shape = env.action_spec().shape
        self.observation_space = gym.spaces.Box(
            low=np.full(obs_shape, -np.inf, dtype=np.float32),
            high=np.full(obs_shape, np.inf, dtype=np.float32),
            dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=np.full(action_shape, env.action_spec().minimum),
            high=np.full(action_shape, env.action_spec().maximum),
            dtype=env.action_spec().dtype)
        self.action_spec_dtype = env.action_spec().dtype
        self.metadata = self._build_metadata()

    @property
    def unwrapped(self):
        return self.env
    
    def _obs_to_array(self, obs):
        return np.concatenate([v.flatten() for v in obs.values()], dtype=np.float32)
    
    def _build_metadata(self):
        control_timestep = self.env.control_timestep()
        render_fps = int(1 / control_timestep)
        return {
            "render_modes": ["human", "rgb_array"],
            "render_fps": render_fps,
            "video.frames_per_second": render_fps 
        }
    
    def reset(self, **kwargs):
        seed = kwargs.get("seed", None)
        if seed is not None:
            if hasattr(self.action_space, 'seed'):
                self.action_space.seed(seed)
            if hasattr(self.observation_space, 'seed'):
                self.observation_space.seed(seed)
        return self._obs_to_array(self.env.reset().observation)

    def step(self, action):
        reward = 0
        action = action.astype(self.action_spec_dtype)
        for _ in range(2):
            step = self.env.step(action)
            reward += step.reward
        return self._obs_to_array(step.observation), reward, False, False, defaultdict(float)
    
    def render(self, width=384, height=384, camera_id=None):
        return self.env.physics.render(height, width, camera_id or self.camera_id)
    
    def close(self):
        pass


def make_env(task):
    """
    Make DMControl environment.
    Adapted from https://github.com/facebookresearch/drqv2
    """
    domain, task = task.replace('-', '_').split('_', 1)
    domain = dict(cup='ball_in_cup', pointmass='point_mass').get(domain, domain)
    if (domain, task) not in suite.ALL_TASKS:
        raise ValueError('Unknown task:', task)
    # assert cfg.obs in {'state', 'rgb'}, 'This task only supports state and rgb observations.'
    env = suite.load(domain,
                     task,
                     task_kwargs={'random': 1},
                     visualize_reward=False)
    env = action_scale.Wrapper(env, minimum=-1., maximum=1.)
    env = DMControlWrapper(env, domain)
    env = Timeout(env, max_episode_steps=500)
    return env