#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Distributed Soft Actor-Critic (DSAC) algorithm
#  Reference: Duan J, Guan Y, Li S E, et al.
#             Distributional soft actor-critic: Off-policy reinforcement learning
#             for addressing value estimation errors[J].
#             IEEE transactions on neural networks and learning systems, 2021.
#  Update: 2021-03-05, Ziqing Gu: create DSAC algorithm
#  Update: 2021-03-05, Wenxuan Wang: debug DSAC algorithm

__all__=["ApproxContainer","DSACU"]
import math
import time
from copy import deepcopy
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam, SGD

from gops.algorithm.base import AlgorithmBase, ApprBase
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.tensorboard_setup import tb_tags
from gops.utils.gops_typing import DataDict
from gops.utils.common_utils import get_apprfunc_dict


class ApproxContainer(ApprBase):
    """Approximate function container for DSAC.

    Contains one policy and multiple action value networks.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        q_args = get_apprfunc_dict("value", **kwargs)
        self.num_q = kwargs["num_q"]
        self.beta_init = kwargs['beta']
        
        for i in range(self.num_q):  
            q_name = f"q{i+1}"
            q_target_name = f"{q_name}_target"
            tmp_q: nn.Module = create_apprfunc(**q_args)
            setattr(self, q_name, tmp_q)
            setattr(self, q_target_name, deepcopy(tmp_q))

        policy_args = get_apprfunc_dict("policy", **kwargs)
        self.policy: nn.Module = create_apprfunc(**policy_args)
        self.policy_target = deepcopy(self.policy)
        if self.lambda_upper:
            self.behavior_policy = deepcopy(self.policy)

        for p in self.policy_target.parameters():
            p.requires_grad = False
        for i in range(self.num_q):  
            q_target_name = f"q{i+1}_target"
            q_target = getattr(self, q_target_name)
            for p in q_target.parameters():
                p.requires_grad = False

        self.log_alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(self.beta_init, dtype=torch.float32))

        for i in range(self.num_q):  
            q_name = f"q{i+1}"
            q_optimizer_name = f"{q_name}_optimizer"
            q_network = getattr(self, q_name)
            setattr(self, q_optimizer_name, Adam(q_network.parameters(), lr=kwargs["value_learning_rate"]))
        
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=kwargs["policy_learning_rate"]
        )
        if self.lambda_upper:
            self.behavior_policy_optimizer = Adam(
                self.behavior_policy.parameters(), lr=kwargs["policy_learning_rate"]
            )
        self.alpha_optimizer = Adam([self.log_alpha], lr=kwargs["alpha_learning_rate"])
        self.beta_optimizer = Adam([self.beta], lr=kwargs["beta_learning_rate"])

    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)
    
    def create_behavior_action_distributions(self, logits):
        return self.behavior_policy.get_act_dist(logits)


class DSACU(AlgorithmBase):
    """DSAC algorithm with three refinements, higher performance and more stable.

    Paper: https://arxiv.org/abs/2310.05858

    :param float gamma: discount factor.
    :param float tau: param for soft update of target network.
    :param float alpha: initial temperature.
    :param bool auto_alpha: whether to adjust temperature automatically.
    :param Optional[float] target_entropy: target entropy for automatic
    :param float delay_update: delay update steps for actor.
        temperature adjustment.
    """

    def __init__(
        self,
        index: int = 0,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = math.e,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        delay_update: int = 2,
        **kwargs: Any,
    ):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.networks.log_alpha.data.fill_(math.log(alpha))
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        if target_entropy is None:
            target_entropy = -kwargs["action_dim"]
        self.target_entropy = target_entropy
        self.delay_update = delay_update
        self.mean_stds = [None] * self.networks.num_q
        self.mean_uncertainty = None
        self.mean_rel_u_epi = None
        self.tau_b = kwargs.get("tau_b", self.tau)
        self.beta_init = kwargs['beta']
        self.max_iteration = kwargs['max_iteration']
        self.lambda_lower = kwargs["lambda_lower"]
        self.lambda_upper = kwargs["lambda_upper"]
        self.share_target = kwargs['share_target']

    @property
    def adjustable_parameters(self):
        return (
            "gamma",
            "tau",
            "auto_alpha",
            "alpha",
            "delay_update",
        )

    def local_update(
            self, 
            data: DataDict, 
            iteration: int, 
            update_beta: bool, 
            overestimation: float
        ) -> dict:
        update_beta = False if overestimation > 0 else True
        tb_info = self._compute_gradient(data, iteration, update_beta, overestimation)
        self._update(iteration, update_beta)
        return tb_info

    def get_remote_update_info(
        self, data: DataDict, iteration: int
    ) -> Tuple[dict, dict]:
        tb_info = self._compute_gradient(data, iteration)

        update_info = {
            "q_grads": [],
            "policy_grad": [p._grad for p in self.networks.policy.parameters()],
            "iteration": iteration,
        }
        
        for i in range(self.networks.num_q):
            q_name = f"q{i+1}"
            q_network = getattr(self.networks, q_name)
            update_info["q_grads"].append([p._grad for p in q_network.parameters()])
            
        if self.auto_alpha:
            update_info.update({"log_alpha_grad":self.networks.log_alpha.grad})

        return tb_info, update_info

    def remote_update(self, update_info: dict):
        iteration = update_info["iteration"]
        q_grads = update_info["q_grads"]
        policy_grad = update_info["policy_grad"]

        for i in range(self.networks.num_q):
            q_name = f"q{i+1}"
            q_network = getattr(self.networks, q_name)
            for p, grad in zip(q_network.parameters(), q_grads[i]):
                p._grad = grad
                
        for p, grad in zip(self.networks.policy.parameters(), policy_grad):
            p._grad = grad
            
        if self.auto_alpha:
            self.networks.log_alpha._grad = update_info["log_alpha_grad"]

        self._update(iteration)

    def _get_alpha(self, requires_grad: bool = False):
        alpha = self.networks.log_alpha.exp()
        if requires_grad:
            return alpha
        else:
            return alpha.item()
        
    def _get_beta(self, requires_grad: bool = False):
        if self.networks.beta.item() < 0:
            with torch.no_grad():  
                self.networks.beta.data.fill_(0.0)
        beta = self.networks.beta
        if requires_grad:
            return beta
        else:
            return beta.item()

    def _compute_gradient(
            self, 
            data: DataDict, 
            iteration: int, 
            update_beta, 
            overestimation, 
        ):
        start_time = time.time()

        obs = data["obs"]
        logits = self.networks.policy(obs)
        logits_mean, logits_std = torch.chunk(logits, chunks=2, dim=-1)
        policy_mean = torch.tanh(logits_mean).mean().item()
        policy_std = logits_std.mean().item()

        act_dist = self.networks.create_action_distributions(logits)
        new_act, new_log_prob = act_dist.rsample()
        data.update({"new_act": new_act, "new_log_prob": new_log_prob})

        if self.lambda_upper:
            behavior_logits = self.networks.behavior_policy(obs)
            behavior_act_dist = self.networks.create_behavior_action_distributions(behavior_logits)
            new_behavior_act, new_behavior_log_prob = behavior_act_dist.rsample()
            data.update({"new_behavior_act": new_behavior_act, "new_behavior_log_prob": new_behavior_log_prob})

        for i in range(self.networks.num_q):
            q_optimizer_name = f"q{i+1}_optimizer"
            q_optimizer = getattr(self.networks, q_optimizer_name)
            q_optimizer.zero_grad()
            
        loss_q, q_values, q_stds, min_stds = self._compute_loss_q(data)
        loss_q.backward()

        for i in range(self.networks.num_q):
            q_name = f"q{i+1}"
            q_network = getattr(self.networks, q_name)
            for p in q_network.parameters():
                p.requires_grad = False

        self.networks.policy_optimizer.zero_grad()
        loss_policy, entropy = self._compute_loss_policy(data)
        loss_policy.backward()

        if self.lambda_upper:
            self.networks.behavior_policy_optimizer.zero_grad()
            loss_behavior_policy = self._compute_loss_behavior_policy(data)
            loss_behavior_policy.backward()

        for i in range(self.networks.num_q):
            q_name = f"q{i+1}"
            q_network = getattr(self.networks, q_name)
            for p in q_network.parameters():
                p.requires_grad = True

        if self.auto_alpha:
            self.networks.alpha_optimizer.zero_grad()
            loss_alpha = self._compute_loss_alpha(data)
            loss_alpha.backward()

        if update_beta:
            mean_uncertainty = max(self.mean_uncertainty, 0.1)
            overestimation = overestimation / mean_uncertainty
            self.networks.beta_optimizer.zero_grad()
            loss_beta = self._compute_loss_beta(overestimation)
            loss_beta.backward()

        tb_info = {
            tb_tags["loss_actor"]: loss_policy.item(),
            tb_tags["loss_critic"]: loss_q.item(),
            "DSAC2/policy_mean-RL iter": policy_mean,
            "DSAC2/policy_std-RL iter": policy_std,
            "DSAC2/entropy-RL iter": entropy.item(),
            "DSAC2/alpha-RL iter": self._get_alpha(),
            "DSAC2/beta-RL iter": self._get_beta(),
            "DSAC2/mean_stds": sum(self.mean_stds) / self.networks.num_q if all(std is not None for std in self.mean_stds) else 0,
            tb_tags["alg_time"]: (time.time() - start_time) * 1000,
        }
        
        q_values_tensor = torch.tensor(q_values)
        q_stds_tensor = torch.tensor(q_stds)

        tb_info.update({
            "DSAC2/critic_avg_q_mean-RL iter": q_values_tensor.mean().item(),
            "DSAC2/critic_avg_q_std-RL iter": q_values_tensor.std().item(),
            "DSAC2/critic_avg_q_range-RL iter": (q_values_tensor.max() - q_values_tensor.min()).item(),
        })

        tb_info.update({
            "DSAC2/critic_avg_std_mean-RL iter": q_stds_tensor.mean().item(),
            "DSAC2/critic_avg_std_std-RL iter": q_stds_tensor.std().item(),
            "DSAC2/critic_avg_std_range-RL iter": (q_stds_tensor.max() - q_stds_tensor.min()).item(),
        })

        valid_mean_stds = [std for std in self.mean_stds if std is not None]
        if valid_mean_stds:
            mean_stds_tensor = torch.tensor(valid_mean_stds)
            tb_info.update({
                "DSAC2/mean_std_mean": mean_stds_tensor.mean().item(),
                "DSAC2/mean_std_std": mean_stds_tensor.std().item(),
                "DSAC2/mean_std_range": (mean_stds_tensor.max() - mean_stds_tensor.min()).item(),
            })

        return tb_info

    def _q_evaluate(self, obs, act, qnet):
        StochaQ = qnet(obs, act)
        mean, std = StochaQ[..., 0], StochaQ[..., -1]
        normal = Normal(torch.zeros_like(mean), torch.ones_like(std))
        z = normal.sample()
        z = torch.clamp(z, -3, 3)
        q_value = mean + torch.mul(z, std)
        return mean, std, q_value

    def _compute_loss_q(self, data: DataDict):
        obs, act, rew, obs2, done = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )
        logits_2 = self.networks.policy_target(obs2)
        act2_dist = self.networks.create_action_distributions(logits_2)
        act2, log_prob_act2 = act2_dist.rsample()

        q_means = []
        q_stds = []
        q_samples = []
        
        for i in range(self.networks.num_q):
            q_name = f"q{i+1}"
            q_network = getattr(self.networks, q_name)
            q_mean, q_std, q_sample = self._q_evaluate(obs, act, q_network)
            q_means.append(q_mean)
            q_stds.append(q_std)
            q_samples.append(q_sample)
            
            if self.mean_stds[i] is None:
                self.mean_stds[i] = torch.mean(q_std.detach())
            else:
                self.mean_stds[i] = (1 - self.tau_b) * self.mean_stds[i] + self.tau_b * torch.mean(q_std.detach())

        with torch.no_grad():
            q_next_means = []
            q_next_stds = []
            q_next_samples = []
            
            for i in range(self.networks.num_q):
                q_target_name = f"q{i+1}_target"
                q_target_network = getattr(self.networks, q_target_name)
                q_next_mean, q_next_std, q_next_sample = self._q_evaluate(
                    obs2, act2, q_target_network
                )
                q_next_means.append(q_next_mean)
                q_next_stds.append(q_next_std)
                q_next_samples.append(q_next_sample)

            q_next_all = torch.stack(q_next_means)
            u_epistemic = torch.var(q_next_all, dim=0)
            q_next_stds_all = torch.stack(q_next_stds)
            u_aleatoric = torch.mean(q_next_stds_all ** 2, dim=0)
            
            factor_aleatoric = (self._get_beta() / self.lambda_lower) ** 2
            uncertainty = torch.sqrt(torch.clip(u_epistemic + factor_aleatoric * u_aleatoric, min=1e-8))
            q_next = torch.mean(q_next_all, dim=0) - self.lambda_lower * uncertainty

            if self.mean_uncertainty is None:
                self.mean_uncertainty = self.lambda_lower * torch.mean(uncertainty.detach())
            else:
                self.mean_uncertainty = (1 - self.tau_b) * self.mean_uncertainty + \
                                        self.lambda_lower * self.tau_b * torch.mean(uncertainty.detach())
                
            if self.share_target:
                distances = torch.abs(q_next_all - q_next.unsqueeze(0))
                avg_distances = torch.mean(distances, dim=1)  # [num_q]
                closest_idx = torch.argmin(avg_distances).item()
                shared_q_next_sample = q_next_samples[closest_idx]

        total_loss = 0
        q_values = []
        q_std_values = []
        min_stds = []
        
        for i in range(self.networks.num_q):
            target_q, target_q_bound = self._compute_target_q(
                rew,
                done,
                q_means[i].detach(),
                self.mean_stds[i].detach(),
                q_next.detach(),
                shared_q_next_sample if self.share_target else q_next_samples[i].detach(),
                log_prob_act2.detach(),
            )
            
            q_std_detach = torch.clamp(q_stds[i], min=0.).detach()
            bias = 0.1

            q_loss = (torch.pow(self.mean_stds[i], 2) + bias) * torch.mean(
                -(target_q - q_means[i]).detach() / (torch.pow(q_std_detach, 2) + bias) * q_means[i]
                - ((torch.pow(q_means[i].detach() - target_q_bound, 2) - q_std_detach.pow(2)) 
                   / (torch.pow(q_std_detach, 3) + bias)) * q_stds[i]
            )

            total_loss += q_loss
            q_values.append(q_means[i].detach().mean())
            q_std_values.append(q_stds[i].detach().mean())
            min_stds.append(q_stds[i].min().detach())

        return total_loss, q_values, q_std_values, min_stds

    def _compute_target_q(self, r, done, q, q_std, q_next, q_next_sample, log_prob_a_next):
        target_q = r + (1 - done) * self.gamma * (
            q_next - self._get_alpha() * log_prob_a_next
        )
        target_q_sample = r + (1 - done) * self.gamma * (
            q_next_sample - self._get_alpha() * log_prob_a_next
        )
        td_bound = 3 * q_std
        difference = torch.clamp(target_q_sample - q, -td_bound, td_bound)
        target_q_bound = q + difference
        return target_q.detach(), target_q_bound.detach()

    def _compute_loss_behavior_policy(self, data: DataDict):
        obs, new_act, new_log_prob = data["obs"], data["new_behavior_act"], data["new_behavior_log_prob"]
        
        q_values = []
        q_stds = []
        for i in range(self.networks.num_q):
            q_name = f"q{i+1}"
            q_network = getattr(self.networks, q_name)
            q_mean, q_std, _ = self._q_evaluate(obs, new_act, q_network)
            q_values.append(q_mean)
            q_stds.append(q_std)
        
        q_all = torch.stack(q_values)
        u_epistemic = torch.var(q_all, dim=0)
        q_stds_all = torch.stack(q_stds)
        u_aleatoric = torch.mean(q_stds_all ** 2, dim=0)

        factor_aleatoric = (self._get_beta() / self.lambda_upper) ** 2
        q = torch.mean(q_all, dim=0) + self.lambda_upper * torch.sqrt(
                torch.clip(u_epistemic + factor_aleatoric * u_aleatoric, min=1e-8)
            )

        loss_policy = (self._get_alpha() * new_log_prob - q).mean()
        return loss_policy
    
    def _compute_loss_policy(self, data: DataDict):
        obs, new_act, new_log_prob = data["obs"], data["new_act"], data["new_log_prob"]
        
        q_values = []
        for i in range(self.networks.num_q):
            q_name = f"q{i+1}"
            q_network = getattr(self.networks, q_name)
            q_mean, _, _ = self._q_evaluate(obs, new_act, q_network)
            q_values.append(q_mean)
        
        u_epistemic = torch.var(torch.stack(q_values), dim=0)
        q = torch.mean(torch.stack(q_values), dim=0) - \
            self.lambda_lower * torch.sqrt(torch.clamp(u_epistemic, min=1e-8))

        loss_policy = (self._get_alpha() * new_log_prob - q).mean()
        entropy = -new_log_prob.detach().mean()
        return loss_policy, entropy

    def _compute_loss_alpha(self, data: DataDict):
        new_log_prob = data["new_log_prob"]
        loss_alpha = (
            -self.networks.log_alpha
            * (new_log_prob.detach() + self.target_entropy).mean()
        )
        return loss_alpha
    
    def _compute_loss_beta(self, overestimation):
        '''delta_beta = learning_rate * overestimation'''
        loss_beta = (
            -self.networks.beta * overestimation
        )
        return loss_beta

    def _update(self, iteration: int, update_beta: bool):
        for i in range(self.networks.num_q):
            q_optimizer_name = f"q{i+1}_optimizer"
            q_optimizer = getattr(self.networks, q_optimizer_name)
            q_optimizer.step()

        if iteration % 10000 == 0:
            print("beta: ", self._get_beta())
        
        if update_beta:
            self.networks.beta_optimizer.step()

        if iteration % self.delay_update == 0:
            self.networks.policy_optimizer.step()
            self.networks.behavior_policy_optimizer.step()

            if self.auto_alpha:
                self.networks.alpha_optimizer.step()

            with torch.no_grad():
                polyak = 1 - self.tau
                for i in range(self.networks.num_q):
                    q_name = f"q{i+1}"
                    q_target_name = f"{q_name}_target"
                    q_network = getattr(self.networks, q_name)
                    q_target_network = getattr(self.networks, q_target_name)
                    for p, p_targ in zip(
                        q_network.parameters(), q_target_network.parameters()
                    ):
                        p_targ.data.mul_(polyak)
                        p_targ.data.add_((1 - polyak) * p.data)
                
                for p, p_targ in zip(
                    self.networks.policy.parameters(),
                    self.networks.policy_target.parameters(),
                ):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)