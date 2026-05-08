__all__=["ApproxContainer","DSACAID"]
import math
import time
from copy import deepcopy
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.functional import huber_loss
from torch.distributions import Normal
from torch.optim import Adam

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
        self.lambda_upper = kwargs['lambda_upper']
        
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
        self.beta_optimizer = Adam([self.beta], lr=kwargs["beta_annealing_rate"])

    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)
    
    def create_behavior_action_distributions(self, logits):
        return self.behavior_policy.get_act_dist(logits)


class DSACAID(AlgorithmBase):
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
            target_entropy = -kwargs["entropy_scale_ratio"] * kwargs["action_dim"]
        self.target_entropy = target_entropy
        self.delay_update = delay_update
        self.mean_sigmas = [None] * self.networks.num_q
        self.mean_uncertainty = None
        self.mean_u_epistemic = None
        self.tau_b = kwargs.get("tau_b", self.tau)
        self.beta_init = kwargs['beta']
        self.max_iteration = kwargs['max_iteration']
        self.lambda_lower = kwargs["lambda_lower"]
        self.lambda_upper = kwargs["lambda_upper"]
        self.enable_epi_step_scale = kwargs["enable_epi_step_scale"]
        self.use_homogeneous_sigma_step_ratio = kwargs["use_homogeneous_sigma_step_ratio"]
        self.use_huber_loss = kwargs['use_huber_loss']
        self.q_delta = kwargs.get('q_delta_in_huber_loss', 50.0)
        self.sigma_delta = kwargs.get('sigma_delta_in_sigma_loss', 50.0)
        self.q_bias_lower_threshold = kwargs["q_bias_lower_threshold"]

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
        if overestimation > 0:
            update_beta = False
        tb_info = self._compute_gradient(data, update_beta, overestimation)
        self._update(iteration, update_beta)
        return tb_info

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
            
        loss_q, qs_tensor, sigmas_tensor = self._compute_loss_q(data)
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
            overestimation = max(overestimation, self.q_bias_lower_threshold) 
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
            "DSAC2/mean_sigmas": sum(self.mean_sigmas) / self.networks.num_q if all(std is not None for std in self.mean_sigmas) else 0,
            tb_tags["alg_time"]: (time.time() - start_time) * 1000,
            "DSAC2/critic_mean_q-RL iter": qs_tensor.mean().item(),
            "DSAC2/critic_ensemble_std_q-RL iter": qs_tensor.std(0).mean().item(),
            "DSAC2/critic_batch_std_q-RL iter": qs_tensor.mean(0).std().item(),
            "DSAC2/critic_mean_sigma-RL iter": sigmas_tensor.mean().item(),
            "DSAC2/critic_ensemble_std_sigma-RL iter": sigmas_tensor.std(0).mean().item(),
            "DSAC2/critic_batch_std_sigma-RL iter": sigmas_tensor.mean(0).std().item(),
        }

        return tb_info

    def _q_evaluate(self, obs, act, qnet):
        StochaQ = qnet(obs, act)
        qs, sigmas = StochaQ[..., 0], StochaQ[..., -1]
        normal = Normal(torch.zeros_like(qs), torch.ones_like(sigmas))
        noise = normal.sample()
        noise = torch.clamp(noise, -3, 3)
        zs = qs + torch.mul(noise, sigmas)
        return qs, sigmas, zs

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

        qs = []; sigmas = []; zs = []
        for i in range(self.networks.num_q):
            q_name = f"q{i+1}"
            q_network = getattr(self.networks, q_name)
            q, sigma, z = self._q_evaluate(obs, act, q_network)
            qs.append(q); sigmas.append(sigma); zs.append(z)
            
            if self.mean_sigmas[i] is None:
                self.mean_sigmas[i] = torch.mean(sigma.detach())
            else:
                self.mean_sigmas[i] = (1 - self.tau_b) * self.mean_sigmas[i] + \
                    self.tau_b * torch.mean(sigma.detach())

        q_tensor = torch.stack(qs).detach()
        sigmas_tensor = torch.stack(sigmas).detach()
        u_epistemic = torch.var(q_tensor, dim=0)
        if self.mean_u_epistemic is None:
            self.mean_u_epistemic = torch.mean(u_epistemic)
        else:
            self.mean_u_epistemic = (1 - self.tau_b) * self.mean_u_epistemic + \
                self.tau_b * torch.mean(u_epistemic)

        with torch.no_grad():
            qs_next = []; sigmas_next = []; zs_next = [] 
            for i in range(self.networks.num_q):
                q_target_name = f"q{i+1}_target"
                q_target_network = getattr(self.networks, q_target_name)
                q_next, sigma_next, z_next = self._q_evaluate(
                    obs2, act2, q_target_network
                )
                qs_next.append(q_next)
                sigmas_next.append(sigma_next)
                zs_next.append(z_next)

            q_next_tensor = torch.stack(qs_next)
            u_next_epistemic = torch.var(q_next_tensor, dim=0)
            sigma_next_tensor = torch.stack(sigmas_next)
            u_next_aleatoric = torch.mean(sigma_next_tensor ** 2, dim=0)
            
            factor_aleatoric = (self._get_beta() / self.lambda_lower) ** 2
            uncertainty = torch.sqrt(torch.clip(u_next_epistemic + factor_aleatoric * u_next_aleatoric, min=1e-8))
            target_q_next = torch.mean(q_next_tensor, dim=0) - self.lambda_lower * uncertainty

            if self.mean_uncertainty is None:
                self.mean_uncertainty = self.lambda_lower * torch.mean(uncertainty.detach())
            else:
                self.mean_uncertainty = (1 - self.tau_b) * self.mean_uncertainty + \
                    self.lambda_lower * self.tau_b * torch.mean(uncertainty.detach())
        
        total_loss = 0; bias = 0.1
        for i in range(self.networks.num_q):
            target_q, target_z_bound = self._compute_target_q(
                rew,
                done,
                qs[i].detach(),
                self.mean_sigmas[i].detach(),
                target_q_next.detach(),
                zs_next[i].detach(),
                log_prob_act2.detach(),
            )
            
            sigma_detach = torch.clamp(sigmas[i], min=0.).detach()
            if self.enable_epi_step_scale:
                q_ratio = (
                    (torch.pow(self.mean_sigmas[i], 2) + self.mean_u_epistemic + bias) / \
                    (torch.pow(sigma_detach, 2) + u_epistemic + bias)
                )
            else:
                q_ratio = (
                    (torch.pow(self.mean_sigmas[i], 2) + bias) / \
                    (torch.pow(sigma_detach, 2) + bias)
                )
            if self.use_homogeneous_sigma_step_ratio:
                sigma_ratio = (
                    (torch.pow(self.mean_sigmas[i], 2) + bias) / \
                    (torch.pow(sigma_detach, 2) + bias)
                )
            else:
                sigma_ratio = (
                    (torch.pow(self.mean_sigmas[i], 2) + bias) / \
                    (torch.pow(sigma_detach, 3) + bias)
                )

            if self.use_huber_loss:
                assert self.use_homogeneous_sigma_step_ratio is True, (
                    f"huber loss only supports use_homogeneous_sigma_step_ratio being True! "
                )
                q_ratio = q_ratio.clamp(min=0.1, max=10)
                sigma_ratio = sigma_ratio.clamp(min=0.01, max=100)
                
                residual_value = torch.pow(qs[i].detach()-target_z_bound, 2)
                q_loss = torch.mean(
                    q_ratio * huber_loss(qs[i], target_q, delta = self.q_delta, reduction='none') + \
                    sigma_ratio * huber_loss(sigmas[i].pow(2), residual_value, delta = self.sigma_delta, reduction='none') 
                        / (2 * sigma_detach.pow(2) + bias)
                )
            else:
                if self.use_homogeneous_sigma_step_ratio:
                    q_loss = - torch.mean(
                        q_ratio * qs[i] * (target_q - qs[i]).detach() + \
                        sigma_ratio * sigmas[i] * (
                            torch.pow(qs[i].detach() - target_z_bound, 2) - sigma_detach.pow(2)
                        ) / (sigma_detach + bias)
                    )
                else:
                    q_loss = - torch.mean(
                        q_ratio * qs[i] * (target_q - qs[i]).detach() + \
                        sigma_ratio * sigmas[i] * (
                            torch.pow(qs[i].detach() - target_z_bound, 2) - sigma_detach.pow(2)
                        )
                    )

            total_loss += q_loss

        return total_loss, q_tensor, sigmas_tensor

    def _compute_target_q(self, r, done, q, sigma, q_next, z_next, log_prob_a_next):
        target_q = r + (1 - done) * self.gamma * (
            q_next - self._get_alpha() * log_prob_a_next
        )
        target_z = r + (1 - done) * self.gamma * (
            z_next - self._get_alpha() * log_prob_a_next
        )
        td_bound = 3 * sigma
        difference = torch.clamp(target_z - q, -td_bound, td_bound)
        target_z_bound = q + difference
        return target_q.detach(), target_z_bound.detach()

    def _compute_loss_behavior_policy(self, data: DataDict):
        obs, new_act, new_log_prob = data["obs"], data["new_behavior_act"], data["new_behavior_log_prob"]
        
        qs = []; sigmas = []
        for i in range(self.networks.num_q):
            q_name = f"q{i+1}"
            q_network = getattr(self.networks, q_name)
            q, sigma, _ = self._q_evaluate(obs, new_act, q_network)
            qs.append(q); sigmas.append(sigma)
        
        q_all = torch.stack(qs)
        u_epistemic = torch.var(q_all, dim=0)
        q_stds_all = torch.stack(sigmas)
        u_aleatoric = torch.mean(q_stds_all ** 2, dim=0)

        factor_aleatoric = (self._get_beta() / self.lambda_upper) ** 2
        target_q = torch.mean(q_all, dim=0) + self.lambda_upper * torch.sqrt(
                torch.clip(u_epistemic + factor_aleatoric * u_aleatoric, min=1e-8)
            )
        loss_policy = (self._get_alpha() * new_log_prob - target_q).mean()
        return loss_policy
    
    def _compute_loss_policy(self, data: DataDict):
        obs, new_act, new_log_prob = data["obs"], data["new_act"], data["new_log_prob"]
        
        qs = []
        for i in range(self.networks.num_q):
            q_name = f"q{i+1}"
            q_network = getattr(self.networks, q_name)
            q, _, _ = self._q_evaluate(obs, new_act, q_network)
            qs.append(q)
        
        u_epistemic = torch.var(torch.stack(qs), dim=0)
        target_q = torch.mean(torch.stack(qs), dim=0) - \
            self.lambda_lower * torch.sqrt(torch.clamp(u_epistemic, min=1e-8))
        loss_policy = (self._get_alpha() * new_log_prob - target_q).mean()
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
            if self.lambda_upper:
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