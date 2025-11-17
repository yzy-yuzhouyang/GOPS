#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Monte Carlo Sampler
#  Update Date: 2021-03-10, Wenhan CAO: Revise Codes
#  Update Date: 2021-03-05, Wenxuan Wang: add action clip
#  Update Date: 2023-07-22, Zhilong Zheng: inherit from BaseSampler


from typing import List
import torch
import numpy as np

from gops.trainer.sampler.base import BaseSampler, Experience


class OffProSampler(BaseSampler):
    def __init__(
        self, 
        sample_batch_size,
        index=0, 
        noise_params=None,
        **kwargs
    ):
        super().__init__(
            sample_batch_size,
            index, 
            noise_params,
            **kwargs
        )
        self.use_target_policy = True
    
    def _sample(self) -> List[Experience]:
        batch_data = []
        for _ in range(self.horizon):
            experiences = self._step()
            batch_data.extend(experiences)
        return batch_data
    
    def get_action(self):
        if not self._is_vector:
            batch_obs = torch.from_numpy(
                np.expand_dims(self.obs, axis=0).astype("float32")
            )
        else:
            batch_obs = torch.from_numpy(self.obs.astype("float32"))
        if self.use_target_policy:
            logits = self.networks.policy(batch_obs)
        else:
            logits = self.networks.behavior_policy(batch_obs)
        action_distribution = self.networks.create_action_distributions(logits)
        action, logp = action_distribution.sample()

        if self._is_vector:
            action = action.detach().numpy()
            logp = logp.detach().numpy()
        else:
            action = action.detach()[0].numpy()
            logp = logp.detach()[0].numpy()

        if self.noise_params is not None:
            action = self.noise_processor.sample(action)
        
        return action, logp