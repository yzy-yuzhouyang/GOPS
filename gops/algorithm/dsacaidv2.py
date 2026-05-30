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
        self.use_higher_order_target = kwargs.get("use_higher_order_target", False)
        
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
            if self.use_higher_order_target:
                q_higher_order_target_name = f"q{i+1}_higher_order_target"
                tmp_q_higher_order_target: nn.Module = deepcopy(tmp_q)
                setattr(self, q_higher_order_target_name, tmp_q_higher_order_target)
                q_higher_order_target = getattr(self, q_higher_order_target_name)
                for p in q_higher_order_target.parameters():
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


class DSACAIDV2(AlgorithmBase):
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
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        delay_update: int = 2,
        **kwargs: Any,
    ):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        alpha_init = kwargs["alpha_init"]
        self.networks.log_alpha.data.fill_(math.log(alpha_init))
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        if target_entropy is None:
            target_entropy = -kwargs["entropy_scale_ratio"] * kwargs["action_dim"]
        self.target_entropy = target_entropy
        self.delay_update = delay_update
        self.mean_sigmas_z = [None] * self.networks.num_q
        self.mean_sigmas_q = [None] * self.networks.num_q
        self.penalty = None
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
        self.use_n_form_sigma_q = kwargs["use_n_form_sigma_q"]
        self.use_higher_order_target = kwargs.get("use_higher_order_target", False)
        self.beta_ratio = kwargs.get("beta_ratio", 1.0)
        self.clip_thd = kwargs.get("clip_thd", 1.0)
        self.asymmetry_thd = kwargs["asymmetry_thd"]
        self.asymmetry_ratio = kwargs["asymmetry_ratio"]
        self.use_higher_order_epi = kwargs.get("use_higher_order_epi", False)

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
        # if overestimation > 0:
        #     update_beta = False
        tb_info = self._compute_gradient(data, update_beta, overestimation)
        self._update(iteration, update_beta)
        return tb_info

    def get_remote_update_info(
        self, data: DataDict, iteration: int
    ) -> Tuple[dict, dict]:
        tb_info = self._compute_gradient(data, update_beta=False, overestimation=0)

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
            
        loss_q, qs_tensor, sigmas_z_tensor, sigmas_q_tensor = self._compute_loss_q(data)
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
        #     overestimation = max(overestimation, self.q_bias_lower_threshold) 
        #     mean_uncertainty = max(self.penalty, 0.1)
        #     overestimation = overestimation / mean_uncertainty
        #     self.networks.beta_optimizer.zero_grad()
        #     loss_beta = self._compute_loss_beta(overestimation)
        #     loss_beta.backward()
            overestimation = max(overestimation, self.q_bias_lower_threshold) 
            mean_penalty = max(self.penalty, 0.01)
            overestimation = overestimation / mean_penalty
            self.networks.beta_optimizer.zero_grad()
            loss_beta = self._compute_loss_beta(overestimation)
            loss_beta.backward()

        tb_info = {
            tb_tags["loss_actor"]: loss_policy.item(),
            tb_tags["loss_critic"]: loss_q.item(),
            "DSAC-AID2-actor/1. policy_mean-RL iter": policy_mean,
            "DSAC-AID2-actor/2. policy_std-RL iter": policy_std,
            "DSAC-AID2-actor/3. entropy-RL iter": entropy.item(),
            "DSAC-AID2-critic/1. alpha-RL iter": self._get_alpha(),
            "DSAC-AID2-critic/2. beta-RL iter": self._get_beta(),
            "DSAC-AID2-critic/3. mean_sigmas_z": sum(self.mean_sigmas_z) / self.networks.num_q if all(std is not None for std in self.mean_sigmas_z) else 0,
            "DSAC-AID2-critic/4. mean_sigmas_q": sum(self.mean_sigmas_q) / self.networks.num_q if all(std is not None for std in self.mean_sigmas_q) else 0,
            tb_tags["alg_time"]: (time.time() - start_time) * 1000,
        }

        sigma_q_z_ratio = torch.clamp(sigmas_q_tensor / (sigmas_z_tensor + 1e-8), max=10.0)
        tb_info.update({
            "DSAC-AID2-critic/5. mean_q-RL iter": qs_tensor.mean().item(),
            "DSAC-AID2-critic/6. ensemble_std_q-RL iter": qs_tensor.std(0).mean().item(),
            "DSAC-AID2-critic/7. batch_std_q-RL iter": qs_tensor.mean(0).std().item(),
            "DSAC-AID2-critic/8. mean_sigma_z-RL iter": sigmas_z_tensor.mean().item(),
            "DSAC-AID2-critic/9. ensemble_std_sigma_z-RL iter": sigmas_z_tensor.std(0).mean().item(),
            "DSAC-AID2-critic/10. batch_std_sigma_z-RL iter": sigmas_z_tensor.mean(0).std().item(),
            "DSAC-AID2-critic/11. mean_sigma_q-RL iter": sigmas_q_tensor.mean().item(),
            "DSAC-AID2-critic/12. ensemble_std_sigma_q-RL iter": sigmas_q_tensor.std(0).mean().item(),
            "DSAC-AID2-critic/13. batch_std_sigma_q-RL iter": sigmas_q_tensor.mean(0).std().item(),
            "DSAC-AID2-critic/14. mean_sigma_q_z_ratio-RL iter": sigma_q_z_ratio.mean().item(),
            "DSAC-AID2-critic/15. ensemble_std_sigma_q_z_ratio-RL iter": sigma_q_z_ratio.std(0).mean().item(),
            "DSAC-AID2-critic/16. batch_std_sigma_q_z_ratio-RL iter": sigma_q_z_ratio.mean(0).std().item(),
        })

        return tb_info

    def _q_evaluate(self, obs, act, qnet):
        StochaQ = qnet(obs, act)
        qs, sigmas_z, sigmas_q = StochaQ[..., 0], StochaQ[..., 1], StochaQ[..., 2]
        if self.use_n_form_sigma_q:
            sigmas_q = sigmas_z.detach() / torch.sqrt(sigmas_q+1)
        normal = Normal(torch.zeros_like(qs), torch.ones_like(sigmas_z))
        noise = normal.sample()
        noise = torch.clamp(noise, -3, 3)
        zs = qs + torch.mul(noise, sigmas_z)
        return qs, zs, sigmas_q, sigmas_z

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

        qs = []; zs = []; sigmas_q = []; sigmas_z = []
        for i in range(self.networks.num_q):
            q_name = f"q{i+1}"
            q_network = getattr(self.networks, q_name)
            q, z, sigma_q, sigma_z = self._q_evaluate(obs, act, q_network)
            qs.append(q); zs.append(z); sigmas_q.append(sigma_q); sigmas_z.append(sigma_z)
            
            if self.mean_sigmas_z[i] is None:
                self.mean_sigmas_z[i] = torch.mean(sigma_z.detach())
                self.mean_sigmas_q[i] = torch.mean(sigma_q.detach())
            else:
                self.mean_sigmas_z[i] = (1 - self.tau_b) * self.mean_sigmas_z[i] + \
                    self.tau_b * torch.mean(sigma_z.detach())
                self.mean_sigmas_q[i] = (1 - self.tau_b) * self.mean_sigmas_q[i] + \
                    self.tau_b * torch.mean(sigma_q.detach())

        with torch.no_grad():
            qs_next = []; target_qs = []; target_qs_next =[]; higher_order_target_qs = []
            zs_next = []; sigmas_q_next = []; sigmas_z_next = []
            for i in range(self.networks.num_q):
                q_target_name = f"q{i+1}_target"
                q_target_network = getattr(self.networks, q_target_name)
                if self.use_higher_order_target:
                    q_next, z_next, sigma_q_next, sigma_z_next = self._q_evaluate(
                        obs2, act2, q_target_network
                    )
                else:
                    q_name = f"q{i+1}"
                    q_network = getattr(self.networks, q_name)
                    q_next, z_next, sigma_q_next, sigma_z_next = self._q_evaluate(
                        obs2, act2, q_network
                    )
                target_q, _, _, _ = self._q_evaluate(
                    obs, act, q_target_network
                )
                if self.use_higher_order_target:
                    q_higher_order_target_name = f"q{i+1}_higher_order_target"
                    q_higher_order_target_network = getattr(self.networks, q_higher_order_target_name)
                    higher_order_target_q, _, _, _ = self._q_evaluate(
                        obs, act, q_higher_order_target_network
                    )
                    higher_order_target_qs.append(higher_order_target_q)
                if self.use_higher_order_epi:
                    q_higher_order_target_name = f"q{i+1}_higher_order_target"
                    q_higher_order_target_network = getattr(self.networks, q_higher_order_target_name)
                    target_q_next, _, _, _ = self._q_evaluate(
                        obs2, act2, q_higher_order_target_network
                    )
                    target_qs_next.append(target_q_next)
                qs_next.append(q_next); target_qs.append(target_q); zs_next.append(z_next)
                sigmas_q_next.append(sigma_q_next); sigmas_z_next.append(sigma_z_next)

            q_tensor = torch.stack(qs).detach()
            q_next_tensor = torch.stack(qs_next)
            sigmas_z_tensor = torch.stack(sigmas_z).detach()
            sigmas_z_next_tensor = torch.stack(sigmas_z_next)
            sigmas_q_tensor = torch.stack(sigmas_q).detach()
            sigmas_q_next_tensor = torch.stack(sigmas_q_next)
            
            if self.networks.num_q <= 1:
                u_epistemic = torch.zeros_like(q_tensor[0])
                u_next_epistemic = torch.zeros_like(q_next_tensor[0])
            elif self.use_higher_order_epi:
                target_q_tensor = torch.stack(target_qs).detach()
                target_q_next_tensor = torch.stack(target_qs_next).detach()
                u_epistemic = torch.var(target_q_tensor, dim=0)
                u_next_epistemic = torch.var(target_q_next_tensor, dim=0)
            else:
                u_epistemic = torch.var(q_tensor, dim=0)
                u_next_epistemic = torch.var(q_next_tensor, dim=0)
            if self.mean_u_epistemic is None:
                self.mean_u_epistemic = torch.mean(u_epistemic)
            else:
                self.mean_u_epistemic = (1 - self.tau_b) * self.mean_u_epistemic + \
                    self.tau_b * torch.mean(u_epistemic)
                
            q_next_aleatoric = torch.mean(sigmas_q_next_tensor ** 2, dim=0)
            z_next_aleatoric = torch.mean(sigmas_z_next_tensor ** 2, dim=0)
            max_q_next_aleatoric = z_next_aleatoric * (self.clip_thd ** 2)
            q_next_aleatoric = torch.clamp(self.beta_ratio * q_next_aleatoric, max=max_q_next_aleatoric)

            uncertainty = torch.sqrt(torch.clip(u_next_epistemic + q_next_aleatoric, min=1e-8))
            penalty = self.lambda_lower * self._get_beta() * uncertainty
            target_q_next = torch.mean(q_next_tensor, dim=0) - penalty

            if self.penalty is None:
                self.penalty = torch.mean(uncertainty.detach())
            else:
                self.penalty = (1 - self.tau_b) * self.penalty + \
                    self.tau_b * torch.mean(uncertainty.detach())
        
        total_loss = 0; bias = 0.1
        for i in range(self.networks.num_q):
            target_q_bellman, target_z_bound = self._compute_target_q(
                rew,
                done,
                qs[i].detach(),
                self.mean_sigmas_z[i].detach(),
                target_q_next.detach(),
                zs_next[i].detach(),
                log_prob_act2.detach(),
            )
            
            sigma_z_detach = torch.clamp(sigmas_z[i], min=0.).detach()
            sigma_q_detach = torch.clamp(sigmas_q[i], min=0.).detach()
            if self.enable_epi_step_scale:
                q_ratio = (
                    (torch.pow(self.mean_sigmas_z[i], 2) + self.mean_u_epistemic + bias) / \
                    (torch.pow(sigma_z_detach, 2) + u_epistemic + bias)
                )
            else:
                q_ratio = (
                    (torch.pow(self.mean_sigmas_z[i], 2) + bias) / \
                    (torch.pow(sigma_z_detach, 2) + bias)
                )
            if self.use_homogeneous_sigma_step_ratio:
                sigma_z_ratio = (
                    (torch.pow(self.mean_sigmas_z[i], 2) + bias) / \
                    (torch.pow(sigma_z_detach, 2) + bias)
                )
                sigma_q_ratio = (
                    (torch.pow(self.mean_sigmas_q[i], 2) + bias) / \
                    (torch.pow(sigma_q_detach, 2) + bias)
                )
            else:
                sigma_z_ratio = (
                    (torch.pow(self.mean_sigmas_z[i], 2) + bias) / \
                    (torch.pow(sigma_z_detach, 3) + bias)
                )
                sigma_q_ratio = (
                    (torch.pow(self.mean_sigmas_q[i], 2) + bias) / \
                    (torch.pow(sigma_q_detach, 3) + bias)
                )

            if self.use_higher_order_target:
                sigma_q_difference = higher_order_target_qs[i].detach() - target_qs[i].detach()
            else:
                sigma_q_difference = target_qs[i].detach() - qs[i].detach()
            if self.use_huber_loss:
                assert self.use_homogeneous_sigma_step_ratio is True, (
                    f"huber loss only supports use_homogeneous_sigma_step_ratio being True! "
                )
                q_ratio = q_ratio.clamp(min=0.1, max=10)
                sigma_z_ratio = sigma_z_ratio.clamp(min=0.01, max=100)
                
                q_loss = torch.mean(
                    q_ratio * huber_loss(qs[i], target_q_bellman, delta = self.q_delta, reduction='none') + \
                    sigma_z_ratio * sigmas_z[i] * (sigma_z_detach.pow(2) - huber_loss(qs[i].detach(), target_z_bound, delta = self.sigma_delta, reduction='none'))
                        / (sigma_z_detach + bias) + \
                    sigma_q_ratio * sigmas_q[i] * (
                            sigma_q_detach.pow(2) - torch.pow(sigma_q_difference, 2)
                        ) / (sigma_q_detach + bias)
                )
                # q_loss = torch.mean(
                #     q_ratio * huber_loss(qs[i], target_q_bellman, delta = self.q_delta, reduction='none') + \
                #     sigma_z_ratio * sigmas_z[i] * (sigma_z_detach.pow(2) - 2 * huber_loss(target_q_bellman, target_z_bound, delta = self.sigma_delta, reduction='none'))
                #         / (sigma_z_detach + bias) + \
                #     sigma_q_ratio * sigmas_q[i] * (
                #             sigma_q_detach.pow(2) - torch.pow(sigma_q_difference, 2)
                #         ) / (sigma_q_detach + bias)
                # )
            else:
                if self.use_homogeneous_sigma_step_ratio:
                    q_loss = - torch.mean(
                        q_ratio * qs[i] * (target_q_bellman - qs[i]).detach() + \
                        sigma_z_ratio * sigmas_z[i] * (
                            torch.pow(qs[i].detach() - target_z_bound, 2) - sigma_z_detach.pow(2)
                        ) / (sigma_z_detach + bias) + \
                        sigma_q_ratio * sigmas_q[i] * (
                            torch.pow(sigma_q_difference, 2) - sigma_q_detach.pow(2)
                        ) / (sigma_q_detach + bias)
                    )
                else:
                    q_loss = - torch.mean(
                        q_ratio * qs[i] * (target_q_bellman - qs[i]).detach() + \
                        sigma_z_ratio * sigmas_z[i] * (
                            torch.pow(qs[i].detach() - target_z_bound, 2) - sigma_z_detach.pow(2)
                        ) + \
                        sigma_q_ratio * sigmas_q[i] * (
                            torch.pow(sigma_q_difference, 2) - sigma_q_detach.pow(2)
                        )
                    )

            total_loss += q_loss

        return total_loss, q_tensor, sigmas_z_tensor, sigmas_q_tensor

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
        
        qs = []; sigmas_q = []; sigmas_z = []; target_qs = []
        for i in range(self.networks.num_q):
            q_name = f"q{i+1}"
            q_network = getattr(self.networks, q_name)
            q, _, sigma_q, sigma_z = self._q_evaluate(obs, new_act, q_network)
            qs.append(q); sigmas_q.append(sigma_q); sigmas_z.append(sigma_z)
            if self.use_higher_order_epi:
                q_target_name = f"q{i+1}_target"
                q_target_network = getattr(self.networks, q_target_name)
                target_q, _, _, _ = self._q_evaluate(obs, new_act, q_target_network)
                target_qs.append(target_q)
        
        q_all = torch.stack(qs)
        if self.networks.num_q <= 1:
            u_epistemic = torch.zeros_like(q_all[0])
        elif self.use_higher_order_epi:
            target_q_all = torch.stack(target_qs)
            u_epistemic = torch.var(target_q_all, dim=0)
        else:
            u_epistemic = torch.var(q_all, dim=0)
        q_stds_all = torch.stack(sigmas_q)
        z_stds_all = torch.stack(sigmas_z)
        q_aleatoric = torch.mean(q_stds_all ** 2, dim=0)
        z_aleatoric = torch.mean(z_stds_all ** 2, dim=0)
        q_z_ratio = q_aleatoric / z_aleatoric
        q_z_ratio = torch.where(
            q_z_ratio > self.asymmetry_thd**2,
            q_z_ratio + (q_z_ratio - self.asymmetry_thd**2) * self.asymmetry_ratio,
            q_z_ratio,
        )
        max_q_aleatoric = z_aleatoric * (self.clip_thd ** 2)
        q_aleatoric = torch.clamp(self.beta_ratio * q_aleatoric, max=max_q_aleatoric)

        target_q = torch.mean(q_all, dim=0) + self.lambda_upper * self._get_beta() * torch.sqrt(
            torch.clip(u_epistemic + q_aleatoric, min=1e-8)
        )

        loss_policy = (self._get_alpha() * new_log_prob - target_q).mean()
        return loss_policy
    
    def _compute_loss_policy(self, data: DataDict):
        obs, new_act, new_log_prob = data["obs"], data["new_act"], data["new_log_prob"]
        
        qs = []; target_qs = []
        for i in range(self.networks.num_q):
            q_name = f"q{i+1}"
            q_network = getattr(self.networks, q_name)
            q, _, _, _ = self._q_evaluate(obs, new_act, q_network)
            qs.append(q)
            if self.use_higher_order_epi:
                q_target_name = f"q{i+1}_target"
                q_target_network = getattr(self.networks, q_target_name)
                target_q, _, _, _ = self._q_evaluate(obs, new_act, q_target_network)
                target_qs.append(target_q)
        
        q_all = torch.stack(qs)
        if self.networks.num_q <= 1:
            u_epistemic = torch.zeros_like(qs[0])
        elif self.use_higher_order_epi:
            target_q_all = torch.stack(target_qs)
            u_epistemic = torch.var(target_q_all, dim=0)
        else:
            u_epistemic = torch.var(q_all, dim=0)
        target_q = torch.mean(q_all, dim=0) - \
            self.lambda_lower * self._get_beta() * torch.sqrt(torch.clamp(u_epistemic, min=1e-8))

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
                    if self.use_higher_order_target:
                        higher_order_q_target_name = f"q{i+1}_higher_order_target"
                        q_higher_order_target_network = getattr(self.networks, higher_order_q_target_name)
                        for p, p_targ, p_h_targ in zip(
                            q_network.parameters(), q_target_network.parameters(), q_higher_order_target_network.parameters()
                        ):
                            p_targ.data.mul_(polyak)
                            p_targ.data.add_((1 - polyak) * p.data)
                            p_h_targ.data.mul_(polyak)
                            p_h_targ.data.add_((1 - polyak) * p_targ.data)
                    else:
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