#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Multilayer Perceptron (MLP)
#  Update: 2021-03-05, Wenjun Zou: create MLP function
#  Update: 2023-07-28, Jiaxin Gao: add FiniteHorizonFullPolicy function
#  Update: 2023-10-25, Wenxuan Wang: add DSAC-T algorithm


__all__ = [
    "DetermPolicy",
    "FiniteHorizonPolicy",
    "FiniteHorizonFullPolicy",
    "MultiplierNet",
    "StochaPolicy",
    "EnhancedStochaPolicy",
    "ActionValue",
    "ActionValueDis",
    "ActionValueDistri",
    "ActionValueEnhancedDistri",
    "StochaPolicyDis",
    "StateValue",
]

import numpy as np
import torch
import warnings
import torch.nn as nn
from gops.utils.common_utils import get_activation_func
from gops.utils.act_distribution_cls import Action_Distribution


# Define MLP function
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


# Count parameter number of MLP
def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def orthogonal_init_(layer, gain=1.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


# Deterministic policy
class DetermPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy.
    Input: observation.
    Output: action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]

        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
            self.pi(obs)
        ) + (self.act_high_lim + self.act_low_lim) / 2
        return action


class FiniteHorizonPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy for finite-horizon.
    Input: observation, time step.
    Output: action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"] + 1
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]

        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs, virtual_t=1):
        virtual_t = virtual_t * torch.ones(
            size=[obs.shape[0], 1], dtype=torch.float32, device=obs.device
        )
        expand_obs = torch.cat((obs, virtual_t), 1)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
            self.pi(expand_obs)
        ) + (self.act_high_lim + self.act_low_lim) / 2
        return action


class MultiplierNet(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy for finite-horizon.
    Input: observation, time step.
    Output: action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"] + 1
        hidden_sizes = kwargs["hidden_sizes"]

        pi_sizes = [obs_dim] + list(hidden_sizes) + [1]
        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )

    def forward(self, obs, virtual_t=1):
        virtual_t = virtual_t * torch.ones(
            size=[obs.shape[0], 1], dtype=torch.float32, device=obs.device
        )
        expand_obs = torch.cat((obs, virtual_t), 1)
        multiplier = self.pi(expand_obs)
        return multiplier
class FiniteHorizonFullPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy for finite-horizon.
    Input: observation, time step.
    Output: action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        self.act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.pre_horizon = kwargs["pre_horizon"]
        pi_sizes = [obs_dim] + list(hidden_sizes) + [self.act_dim * self.pre_horizon]

        self.pi = mlp(
            pi_sizes,
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]).float())
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]).float())
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        return self.forward_all_policy(obs)[:, 0, :]

    def forward_all_policy(self, obs):
        actions = self.pi(obs).reshape(obs.shape[0], self.pre_horizon, self.act_dim)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(actions) \
                 + (self.act_high_lim + self.act_low_lim) / 2
        return action


# Stochastic Policy
class StochaPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of stochastic policy.
    Input: observation.
    Output: parameters of action distribution.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.std_type = kwargs["std_type"]

        # mean and log_std are calculated by different MLP
        if self.std_type == "mlp_separated":
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
            self.mean = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
            self.log_std = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
        # mean and log_std are calculated by same MLP
        elif self.std_type == "mlp_shared":
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim * 2]
            self.policy = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
        # mean is calculated by MLP, and log_std is learnable parameter
        elif self.std_type == "parameter":
            pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
            self.mean = mlp(
                pi_sizes,
                get_activation_func(kwargs["hidden_activation"]),
                get_activation_func(kwargs["output_activation"]),
            )
            self.log_std = nn.Parameter(-0.5*torch.ones(1, act_dim))

        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        if self.std_type == "mlp_separated":
            action_mean = self.mean(obs)
            action_std = torch.clamp(
                self.log_std(obs), self.min_log_std, self.max_log_std
            ).exp()
        elif self.std_type == "mlp_shared":
            logits = self.policy(obs)
            action_mean, action_log_std = torch.chunk(
                logits, chunks=2, dim=-1
            )  # output the mean
            action_std = torch.clamp(
                action_log_std, self.min_log_std, self.max_log_std
            ).exp()
        elif self.std_type == "parameter":
            action_mean = self.mean(obs)
            action_log_std = self.log_std + torch.zeros_like(action_mean)
            action_std = torch.clamp(
                action_log_std, self.min_log_std, self.max_log_std
            ).exp()

        return torch.cat((action_mean, action_std), dim=-1)


class EnhancedStochaPolicy(nn.Module, Action_Distribution):
    """
    基于 DoubleGum 架构优化的随机策略网络
    特点: GroupNorm(no affine), Orthogonal Init, Explicit Heads
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.obs_dim = kwargs["obs_dim"]
        self.act_dim = kwargs["act_dim"]
        self.hidden_sizes = kwargs["hidden_sizes"]
        self.std_type = kwargs["std_type"]
        self.activation = kwargs.get("hidden_activation", "relu")
        
        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))
        self.action_distribution_cls = kwargs["action_distribution_cls"]

        # --- 构建网络 ---
        if self.std_type == "mlp_separated":
            self.mean_backbone = self._build_backbone(self.obs_dim, self.hidden_sizes)
            self.mean_head = nn.Linear(self.hidden_sizes[-1], self.act_dim)
            
            self.log_std_backbone = self._build_backbone(self.obs_dim, self.hidden_sizes)
            self.log_std_head = nn.Linear(self.hidden_sizes[-1], self.act_dim)

        elif self.std_type == "mlp_shared":
            self.backbone = self._build_backbone(self.obs_dim, self.hidden_sizes)
            self.mean_head = nn.Linear(self.hidden_sizes[-1], self.act_dim)
            self.log_std_head = nn.Linear(self.hidden_sizes[-1], self.act_dim)

        elif self.std_type == "parameter":
            self.mean_backbone = self._build_backbone(self.obs_dim, self.hidden_sizes)
            self.mean_head = nn.Linear(self.hidden_sizes[-1], self.act_dim)
            self.log_std = nn.Parameter(-0.5 * torch.ones(1, self.act_dim))

        # --- 初始化 ---
        self.apply(self._init_weights)
        # 显式重置 Head 的初始化为 gain=1.0 (因为 apply 中的 gain 是 sqrt(2))
        self._init_heads()

    def _build_backbone(self, input_dim, hidden_sizes):
        layers = nn.ModuleList()
        curr_dim = input_dim
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(curr_dim, h_dim))
            # GroupNorm (affine=False), 默认16组
            num_groups = 16 if h_dim % 16 == 0 else 1
            layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=h_dim, affine=False))
            # Activation
            if self.activation == "relu":
                layers.append(nn.ReLU())
            elif self.activation == "mish":
                layers.append(nn.Mish())
            else:
                layers.append(nn.ReLU())
            curr_dim = h_dim
        return nn.Sequential(*layers)

    def _init_weights(self, m):
        # 默认所有 Linear 层使用 gain=sqrt(2)
        if isinstance(m, nn.Linear):
            orthogonal_init_(m, gain=np.sqrt(2))

    def _init_heads(self):
        # 将输出头的 gain 重置为 1.0 (DoubleGum 设定)
        heads = []
        if self.std_type == "mlp_separated":
            heads = [self.mean_head, self.log_std_head]
        elif self.std_type == "mlp_shared":
            heads = [self.mean_head, self.log_std_head]
        elif self.std_type == "parameter":
            heads = [self.mean_head]
            
        for head in heads:
            orthogonal_init_(head, gain=1.0)

    def forward(self, obs):
        if self.std_type == "mlp_separated":
            mean_feat = self.mean_backbone(obs)
            action_mean = self.mean_head(mean_feat)
            std_feat = self.log_std_backbone(obs)
            action_log_std = self.log_std_head(std_feat)
            
        elif self.std_type == "mlp_shared":
            feat = self.backbone(obs)
            action_mean = self.mean_head(feat)
            action_log_std = self.log_std_head(feat)
            
        elif self.std_type == "parameter":
            mean_feat = self.mean_backbone(obs)
            action_mean = self.mean_head(mean_feat)
            action_log_std = self.log_std + torch.zeros_like(action_mean)

        action_std = torch.clamp(
            action_log_std, self.min_log_std, self.max_log_std
        ).exp()

        return torch.cat((action_mean, action_std), dim=-1)


class ActionValue(nn.Module, Action_Distribution):
    """
    Approximated function of action-value function.
    Input: observation, action.
    Output: action-value.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.q = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [1],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


class ActionValueDis(nn.Module, Action_Distribution):
    """
    Approximated function of action-value function for discrete action space.
    Input: observation.
    Output: action-value for all action.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_num = kwargs["act_num"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.q = mlp(
            [obs_dim] + list(hidden_sizes) + [act_num],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        return self.q(obs)


class ActionValueDistri(nn.Module):
    """
    Approximated function of distributed action-value function.
    Input: observation.
    Output: parameters of action-value distribution.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.q = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [2],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        if "min_log_std"  in kwargs or "max_log_std" in kwargs:
            warnings.warn("min_log_std and max_log_std are deprecated in ActionValueDistri.")

    def forward(self, obs, act):
        logits = self.q(torch.cat([obs, act], dim=-1))
        value_mean, value_std = torch.chunk(logits, chunks=2, dim=-1)
        value_std = torch.nn.functional.softplus(value_std) 
        
        return torch.cat((value_mean, value_std), dim=-1)


class ActionValueEnhancedDistri(nn.Module):
    """
    基于 DoubleGum 架构优化的分布价值网络 (PyTorch版)
    特点: GroupNorm(no affine), Orthogonal Init, Heteroscedastic Head
    """
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"] # 例如 [256, 256]
        activation = kwargs.get("hidden_activation", "relu") # DoubleGum 论文使用 ReLU
        
        # 构建主干网络 (Backbone)
        self.layers = nn.ModuleList()
        input_dim = obs_dim + act_dim
        
        for h_dim in hidden_sizes:
            self.layers.append(nn.Linear(input_dim, h_dim))
            
            # [DoubleGum 特性 1] GroupNorm, 16组, 关闭 affine (无 shift/scale)
            # 注意：num_channels 必须能被 num_groups 整除。256/16 = 16 (OK)
            self.layers.append(nn.GroupNorm(num_groups=16, num_channels=h_dim, affine=False))
            
            # 激活函数
            if activation == "relu":
                self.layers.append(nn.ReLU())
            elif activation == "mish":
                self.layers.append(nn.Mish())
            else:
                self.layers.append(nn.ReLU()) # 默认
            
            input_dim = h_dim

        # [DoubleGum 特性 3] 独立的输出头，便于分别初始化
        self.mean_head = nn.Linear(input_dim, 1)
        self.std_head = nn.Linear(input_dim, 1)
        
        # [DoubleGum 特性 2] 应用正交初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 隐藏层 gain = sqrt(2)
        if isinstance(m, nn.Linear):
            orthogonal_init_(m, gain=np.sqrt(2))
        # 重新初始化输出头，gain = 1.0 (论文设定)
        orthogonal_init_(self.mean_head, gain=1.0)
        orthogonal_init_(self.std_head, gain=1.0)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        
        for layer in self.layers:
            x = layer(x)
            
        value_mean = self.mean_head(x)
        value_std_logits = self.std_head(x)
        
        # [DoubleGum 特性 3] 数值稳定性修正
        # Softplus 保证正数，+ 1e-5 防止除零或 log(0)
        value_std = torch.nn.functional.softplus(value_std_logits) + 1e-5
        
        return torch.cat((value_mean, value_std), dim=-1)


class StochaPolicyDis(ActionValueDis, Action_Distribution):
    """
    Approximated function of stochastic policy for discrete action space.
    Input: observation.
    Output: parameters of action distribution.
    """

    pass


class StateValue(nn.Module, Action_Distribution):
    """
    Approximated function of state-value function.
    Input: observation, action.
    Output: state-value.
    """

    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.v = mlp(
            [obs_dim] + list(hidden_sizes) + [1],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        v = self.v(obs)
        return torch.squeeze(v, -1)
