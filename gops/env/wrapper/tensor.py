from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch


class TensorWrapper(gym.Wrapper):
    """
    Wrapper for converting numpy arrays to torch tensors.
    """

    def __init__(self, env, reward_scale=1.0):
        super().__init__(env)
        if reward_scale is None:
            self.reward_scale = 1.0
        else:
            self.reward_scale = reward_scale
    
    def rand_act(self):
        return torch.from_numpy(self.action_space.sample().astype(np.float32))

    def _try_f32_tensor(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
            if x.dtype == torch.float64:
                x = x.float()
        return x

    def _obs_to_tensor(self, obs):
        if isinstance(obs, dict):
            for k in obs.keys():
                obs[k] = self._try_f32_tensor(obs[k])
        else:
            obs = self._try_f32_tensor(obs)
        return obs

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple):
            obs, info = out
            info["t0"] = True
        else:
            obs = out
            info = {"t0": True}
        return self._obs_to_tensor(obs), info

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action
        elif isinstance(action, torch.Tensor):
            action = action.numpy()
        out = self.env.step(action)
        obs, reward, te, tr, info = out
        reward_scaled = reward * self.reward_scale
        info['terminated'] = te
        info = defaultdict(float, info)
        info['success'] = float(info['success'])
        info['terminated'] = torch.tensor(float(info['terminated']))
        info['t0'] = False
        return self._obs_to_tensor(obs), torch.tensor(reward_scaled, dtype=torch.float32), te, tr, info