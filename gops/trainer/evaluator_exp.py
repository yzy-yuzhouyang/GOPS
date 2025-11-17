#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Evaluation of trained policy
#  Update Date: 2021-05-10, Yang Guan: renew environment parameters


import numpy as np
import torch
import os

from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_alg import create_approx_contrainer
from gops.utils.common_utils import set_seed


class EvaluatorExp:
    def __init__(self, index=0, **kwargs):
        self.reward_scale = kwargs["reward_scale"]
        kwargs.update({
            "reward_scale": None,
            "repeat_num": None,
            "gym2gymnasium": False,
            "vector_env_num": None,
        })
        self.env = create_env(**kwargs)

        _, self.env = set_seed(kwargs["trainer"], kwargs["seed"], index + 400, self.env)

        self.networks = create_approx_contrainer(**kwargs)
        self.render = kwargs["is_render"]

        self.num_eval_episode = kwargs["num_eval_episode"]
        self.action_type = kwargs["action_type"]
        self.policy_func_name = kwargs["policy_func_name"]
        self.save_folder = kwargs["save_folder"]
        self.eval_save = kwargs.get("eval_save", True)
        self.gamma = kwargs["gamma"]

        self.print_time = 0
        self.print_iteration = -1

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)

    def run_an_episode(self, iteration, render=True):
        if self.print_iteration != iteration:
            self.print_iteration = iteration
            self.print_time = 0
        else:
            self.print_time += 1
        obs_list = []
        action_list = []
        log_prob_list =[]
        reward_list = []
        qx_tb_eval_dict = {}

        obs, info = self.env.reset()
        done = 0
        info["TimeLimit.truncated"] = False
        while not (done or info["TimeLimit.truncated"]):
            batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
            logits = self.networks.policy(batch_obs)
            action_distribution = self.networks.create_action_distributions(logits)
            action = action_distribution.mode()
            log_prob = action_distribution.log_prob(action)
            action = action.detach().numpy()[0]
            log_prob = log_prob.detach().numpy()[0]
            next_obs, reward, done, next_info = self.env.step(action)
            obs_list.append(obs)
            action_list.append(action)
            log_prob_list.append(log_prob)
            obs = next_obs
            info = next_info
            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
            # Draw environment animation
            if render:
                self.env.render()
            reward_list.append(reward)
        eval_dict = {
            "reward_list": reward_list,
            "action_list": action_list,
            "obs_list": obs_list,
        }
        if self.eval_save:
            np.save(
                self.save_folder
                + "/evaluator/iter{}_ep{}".format(iteration, self.print_time),
                eval_dict,
            )

        '''------------------------yzy's part------------------------'''
        Q_true_list = []
        alpha = self.networks.log_alpha.exp().item()
        gamma = self.gamma
        
        with torch.no_grad():
            obs_array = np.array(obs_list[:200], dtype=np.float32)
            action_array = np.array(action_list[:200], dtype=np.float32)
            obs_tensor = torch.from_numpy(obs_array)
            action_tensor = torch.from_numpy(action_array)
            if hasattr(self.networks, 'q1'):
                StochaQ1 = self.networks.q1(obs_tensor, torch.tensor(action_tensor))
                StochaQ2 = self.networks.q2(obs_tensor, torch.tensor(action_tensor))
                Q_output_gpu = torch.min(StochaQ1[..., 0], StochaQ2[..., 0])
                Q_diff_gpu = torch.abs(StochaQ1[..., 0] - StochaQ2[..., 0])
                std_diff_gpu = torch.abs(StochaQ1[..., 1] - StochaQ2[..., 1])
                std_output_gpu = torch.where(
                    StochaQ1[..., 0] < StochaQ2[..., 0],
                    StochaQ1[..., 1], 
                    StochaQ2[..., 1]
                )
                Q_rel_diff_gpu = torch.clip(
                    2 * Q_diff_gpu / torch.clip(torch.abs(StochaQ1[..., 0] + StochaQ2[..., 0]), min=0.1), 
                    max=2.0
                )
                std_rel_diff_gpu = torch.clip(
                    2 * std_diff_gpu / torch.clip(StochaQ1[..., 1] + StochaQ2[..., 1], min=0.1), 
                    max=2.0
                )
            elif hasattr(self.networks, 'q'):
                StochaQ = self.networks.q(obs_tensor, torch.tensor(action_tensor))
                Q_output_gpu = StochaQ[..., 0]
                std_output_gpu = StochaQ[..., 1]
            else:
                raise AttributeError("Neither 'q1' nor 'q' method is found in the networks object. \
                                     Please ensure at least one of them is implemented.")
        Q_output_list = Q_output_gpu.cpu().numpy()
        Q_diff_list = Q_diff_gpu.cpu().numpy()
        std_output_list = std_output_gpu.cpu().numpy()
        std_diff_list = std_diff_gpu.cpu().numpy()
        Q_rel_diff_list = Q_rel_diff_gpu.cpu().numpy()
        std_rel_diff_list = std_rel_diff_gpu.cpu().numpy()
        
        for i in range(min(200, len(reward_list))):
            Q_true = self.reward_scale * (sum([r * (gamma**j) for j, r in enumerate(reward_list[i:-1])]) \
                    - alpha * sum([log_prob * (gamma**j) for j, log_prob in enumerate(log_prob_list[i+1:])]) \
                    + Q_output_list[-1] * (gamma**(len(reward_list)-i-1)))
            Q_true_list.append(Q_true)
        Q_bias_list = Q_output_list - np.array(Q_true_list)
        over_ratio = np.mean(Q_bias_list > 0)

        qx_tb_eval_dict["total_avg_return"] = sum(reward_list)
        qx_tb_eval_dict['real_Q'] = np.mean(Q_true_list)
        qx_tb_eval_dict['output_Q'] = np.mean(Q_output_list)
        qx_tb_eval_dict['Q_diff_mean'] = np.mean(Q_diff_list)
        qx_tb_eval_dict['std_diff_mean'] = np.mean(std_diff_list)
        qx_tb_eval_dict['Q_rel_diff_mean'] = np.mean(Q_rel_diff_list)
        qx_tb_eval_dict['std_rel_diff_mean'] = np.mean(std_rel_diff_list)
        qx_tb_eval_dict['Q_bias_mean'] = np.mean(Q_bias_list)
        qx_tb_eval_dict['Q_bias_std'] = np.std(Q_bias_list)
        qx_tb_eval_dict['Episode_len'] = len(reward_list)
        qx_tb_eval_dict['over_ratio'] = over_ratio

        visualize_dict = {
            "iter": iteration,
            "std": std_output_list,
            "Q": Q_output_list,
            "bias": Q_bias_list,
            "diff": Q_diff_list,
            "std_diff": std_diff_list,
            "rel_diff": Q_rel_diff_list,
            "std_rel_diff": std_rel_diff_list,
        }
        visualize_dir = os.path.join(self.save_folder, "evaluator", "visualize")
        os.makedirs(visualize_dir, exist_ok=True)
        np.save(
            visualize_dir
            + "/iter{}_ep{}".format(iteration, self.print_time),
            visualize_dict,
        )
        '''------------------------yzy's part------------------------'''

        return qx_tb_eval_dict

    def run_n_episodes(self, n, iteration):
        eval_list = []
        for _ in range(n):
            eval_list.append(self.run_an_episode(iteration, self.render)) 
        avg_idsim_tb_eval_dict = {
            k: np.mean([d[k] for d in eval_list]) for k in eval_list[0].keys()
            }
        return avg_idsim_tb_eval_dict

    def run_evaluation(self, iteration):
        return self.run_n_episodes(self.num_eval_episode, iteration)
