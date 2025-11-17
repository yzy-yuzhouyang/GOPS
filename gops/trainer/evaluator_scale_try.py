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


class EvaluatorScaleTry:
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
        self.num_q = self.networks.num_q  # 直接从networks获取Q网络数量

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
            # 确保数据在正确的设备上
            if hasattr(self.networks, 'device'):
                batch_obs = batch_obs.to(self.networks.device)
            logits = self.networks.policy(batch_obs)
            action_distribution = self.networks.create_action_distributions(logits)
            action = action_distribution.mode()
            log_prob = action_distribution.log_prob(action)
            # 将结果移回CPU进行numpy转换
            action = action.cpu().detach().numpy()[0]
            log_prob = log_prob.cpu().detach().numpy()[0]
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
            
            # 确保数据在正确的设备上
            if hasattr(self.networks, 'device'):
                obs_tensor = obs_tensor.to(self.networks.device)
                action_tensor = action_tensor.to(self.networks.device)
            
            # 获取所有Q网络的输出
            all_q_means = []
            all_q_stds = []
            
            for i in range(self.num_q):
                q_name = f"q{i+1}"
                q_network = getattr(self.networks, q_name)
                StochaQ = q_network(obs_tensor, action_tensor)
                all_q_means.append(StochaQ[..., 0])
                all_q_stds.append(StochaQ[..., 1])
            
            # 堆叠所有Q网络的输出
            q_means_tensor = torch.stack(all_q_means)  # [num_q, seq_len]
            q_stds_tensor = torch.stack(all_q_stds)    # [num_q, seq_len]
            
            # 计算Q的置信下界
            q = torch.amin(q_means_tensor, dim=0)
            Q_output_gpu = q
            
            # 标准差
            std_output_gpu = torch.var(q_stds_tensor, dim=0)
            
            # 计算Q值的差异（基于所有Q网络的标准差）
            q_means_var = torch.var(q_means_tensor, dim=0)  # 计算Q值的方差
            Q_diff_gpu = torch.sqrt(q_means_var)  # 标准差作为差异度量
            
            # 计算标准差的差异（基于所有Q网络的标准差）
            q_stds_var = torch.var(q_stds_tensor, dim=0)    # 计算标准差的方差
            std_diff_gpu = torch.sqrt(q_stds_var)           # 标准差作为差异度量
            
            # 计算相对差异
            q_means_mean = torch.mean(q_means_tensor, dim=0)
            q_stds_mean = torch.mean(q_stds_tensor, dim=0)
            
            Q_rel_diff_gpu = torch.clip(
                Q_diff_gpu / torch.clip(torch.abs(q_means_mean), min=0.1), 
                max=2.0
            )
            std_rel_diff_gpu = torch.clip(
                std_diff_gpu / torch.clip(q_stds_mean, min=0.1), 
                max=2.0
            )
            
        # 将结果移回CPU进行numpy转换
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

        # 添加多Q网络的统计信息
        qx_tb_eval_dict['num_q_networks'] = self.num_q
        qx_tb_eval_dict['q_means_std'] = np.std([q_mean.cpu().numpy().mean() for q_mean in all_q_means])
        qx_tb_eval_dict['q_stds_std'] = np.std([q_std.cpu().numpy().mean() for q_std in all_q_stds])

        # visualize_dict = {
        #     "iter": iteration,
        #     "std": std_output_list,
        #     "Q": Q_output_list,
        #     "bias": Q_bias_list,
        #     "diff": Q_diff_list,
        #     "std_diff": std_diff_list,
        #     "rel_diff": Q_rel_diff_list,
        #     "std_rel_diff": std_rel_diff_list,
        #     "num_q": self.num_q,
        #     "all_q_means": [q_mean.cpu().numpy() for q_mean in all_q_means],
        #     "all_q_stds": [q_std.cpu().numpy() for q_std in all_q_stds],
        # }
        # visualize_dir = os.path.join(self.save_folder, "evaluator", "visualize")
        # os.makedirs(visualize_dir, exist_ok=True)
        # np.save(
        #     visualize_dir
        #     + "/iter{}_ep{}".format(iteration, self.print_time),
        #     visualize_dict,
        # )
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