#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Evaluation of trained policy
#  Update Date: 2021-05-10, Yang Guan: renew environment parameters


import numpy as np
import torch

from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_alg import create_approx_contrainer
from gops.utils.common_utils import set_seed


class EvaluatorSac:
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
            if hasattr(self.networks, 'device'):
                batch_obs = batch_obs.to(self.networks.device)
            logits = self.networks.policy(batch_obs)
            action_distribution = self.networks.create_action_distributions(logits)
            action = action_distribution.mode()
            log_prob = action_distribution.log_prob(action)
            action = action.cpu().detach().numpy()[0]
            log_prob = log_prob.cpu().detach().numpy()[0]
            
            out = self.env.step(action)
            import gymnasium
            if isinstance(self.env.action_space, gymnasium.spaces.Space):
                next_obs, reward, te, tr, next_info = out
                done = te or tr
            else:
                next_obs, reward, done, next_info = out

            obs_list.append(obs)
            action_list.append(action)
            log_prob_list.append(log_prob)
            obs = next_obs
            info = next_info
            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
            # Draw environment animation.
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


        # DSAC-T: Compute value statistics.
        with torch.no_grad():
            selected_obs_list = obs_list[:200] + (obs_list[-1:] if len(obs_list) > 200 else [])
            selected_act_list = action_list[:200] + (action_list[-1:] if len(action_list) > 200 else [])
            # Convert recent history to tensors (limit to 200 steps for stability)
            obs_tensor = torch.from_numpy(np.array(selected_obs_list, dtype=np.float32))
            action_tensor = torch.from_numpy(np.array(selected_act_list, dtype=np.float32))
            if hasattr(self.networks, 'device'):
                obs_tensor = obs_tensor.to(self.networks.device)
                action_tensor = action_tensor.to(self.networks.device)
                
            # Collect Q-values from the Q-ensemble.
            output_qs = []
            output_sigmas = []
            for i in range(2):
                q_name = f"q{i+1}"
                q_network = getattr(self.networks, q_name)
                output_qs.append(q_network(obs_tensor, action_tensor))
            output_qs = torch.stack(output_qs)  # [num_q, seq_len]
            mean_output_qs = torch.amin(output_qs, dim=0)
            std_output_qs = torch.sqrt(torch.var(output_qs, dim=0))
            relative_std_output_qs = torch.clip(
                std_output_qs / torch.clip(torch.abs(mean_output_qs), min=0.1), 
                max=2.0
            )
            
        # Move results to CPU for numpy conversion
        mean_output_qs = mean_output_qs.cpu().numpy()
        std_output_qs = std_output_qs.cpu().numpy()
        relative_std_output_qs = relative_std_output_qs.cpu().numpy()
        
        # Compute true Q-values using Monte Carlo returns
        N = min(200, len(reward_list) - 1)
        alpha = self.networks.log_alpha.exp().item()
        gamma = self.gamma
        true_q_list = []
        next_value = mean_output_qs[-1] - alpha * log_prob_list[-1]
        
        iter_range = range(len(reward_list) - 2, -1, -1)
        for i in iter_range:
            r = reward_list[i] * self.reward_scale
            true_q = r + gamma * next_value
            true_q_list.append(true_q)
            current_log_prob = log_prob_list[i]
            next_value = true_q - alpha * current_log_prob

        true_q_list.reverse()
        true_qs = np.array(true_q_list)

        # Compute Q-value bias statistics
        mean_output_qs_aligned = mean_output_qs[:N]
        true_qs_aligned = true_qs[:N]
        assert len(mean_output_qs_aligned) == len(true_qs_aligned), \
            f"Q-value length mismatch: predicted={len(mean_output_qs_aligned)}, true={len(true_qs_aligned)}"         
        q_bias = mean_output_qs_aligned - true_qs_aligned

        # Populate evaluation metrics dictionary
        qx_tb_eval_dict["total_avg_return"] = sum(reward_list)  # Total episode return
        qx_tb_eval_dict['true_q'] = np.mean(true_qs_aligned)  # Average true Q-value
        qx_tb_eval_dict['mean_output_q'] = np.mean(mean_output_qs[:N])  # Average estimated Q-value
        qx_tb_eval_dict['std_output_q'] = np.mean(std_output_qs[:N])  # Mean Q-value disagreement
        qx_tb_eval_dict['relative_std_output_q'] = np.mean(relative_std_output_qs[:N])  # Relative Q disagreement
        qx_tb_eval_dict['mean_q_bias'] = np.mean(q_bias)  # Mean Q-value bias
        qx_tb_eval_dict['std_q_bias'] = np.std(q_bias)  # Std of Q-value bias
        qx_tb_eval_dict['episode_length'] = len(reward_list)  # Episode length
        qx_tb_eval_dict['overestimation_ratio'] = np.mean(q_bias > 0)  # Overestimation ratio

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