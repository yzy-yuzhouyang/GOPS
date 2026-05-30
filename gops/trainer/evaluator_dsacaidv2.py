import numpy as np
import torch
import gymnasium

from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_alg import create_approx_contrainer
from gops.utils.common_utils import set_seed


class EvaluatorDsacaidv2:
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
        # DSAC-AID.
        self.num_q = self.networks.num_q
        self.lambda_lower = kwargs["lambda_lower"]
        self.use_higher_order_epi = kwargs.get("use_higher_order_epi", False)

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
        tb_eval_dict = {}

        obs, info = self.env.reset()
        done = 0
        info["TimeLimit.truncated"] = False
        with torch.no_grad():
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


        # DSAC-AID: Compute value statistics with uncertainty estimation.
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
            output_qs = []; output_sigmas = []; target_qs_next = []
            for i in range(self.num_q):
                q_name = f"q{i+1}"
                q_network = getattr(self.networks, q_name)
                StochaQ = q_network(obs_tensor, action_tensor)
                output_qs.append(StochaQ[..., 0])
                output_sigmas.append(StochaQ[..., 1])
                if self.use_higher_order_epi:
                    q_target_name = f"q{i+1}_target"
                    q_target_network = getattr(self.networks, q_target_name)
                    target_StochaQ = q_target_network(obs_tensor, action_tensor)
                    target_q = target_StochaQ[..., 0]
                    target_qs_next.append(target_q)
            output_qs = torch.stack(output_qs)  # [num_q, seq_len]
            output_sigmas = torch.stack(output_sigmas)    # [num_q, seq_len]

            # Compute final Q-value with uncertainty penalty.
            if self.num_q <= 1:
                u_epistemic = torch.zeros_like(output_qs[0])
            elif self.use_higher_order_epi:
                target_qs_next_tensor = torch.stack(target_qs_next)
                u_epistemic = torch.var(target_qs_next_tensor, dim=0)
            else:
                u_epistemic = torch.var(output_qs, dim=0)
            beta = self.networks.beta.item()
            mean_output_qs = torch.mean(output_qs, dim=0) - \
                self.lambda_lower * torch.sqrt(u_epistemic)
            
            # Compute statistics.
            mean_output_sigmas = torch.mean(output_sigmas, dim=0)
            std_output_qs = torch.sqrt(torch.var(output_qs, dim=0))
            std_output_sigmas = torch.sqrt(torch.var(output_sigmas, dim=0))
            relative_std_output_qs = torch.clip(
                std_output_qs / torch.clip(torch.abs(mean_output_qs), min=0.1), 
                max=2.0
            )
            relative_std_output_sigmas = torch.clip(
                std_output_sigmas / torch.clip(mean_output_sigmas, min=0.1), 
                max=2.0
            )
            
        # Move results to CPU for numpy conversion
        mean_output_qs = mean_output_qs.cpu().numpy()
        std_output_qs = std_output_qs.cpu().numpy()
        std_output_sigmas = std_output_sigmas.cpu().numpy()
        relative_std_output_qs = relative_std_output_qs.cpu().numpy()
        relative_std_output_sigmas = relative_std_output_sigmas.cpu().numpy()
        
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
        tb_eval_dict["total_avg_return"] = sum(reward_list)  # Total episode return
        tb_eval_dict['true_q'] = np.mean(true_qs_aligned)  # Average true Q-value
        tb_eval_dict['mean_output_q'] = np.mean(mean_output_qs[:N])  # Average estimated Q-value
        tb_eval_dict['std_output_q'] = np.mean(std_output_qs[:N])  # Mean Q-value disagreement
        tb_eval_dict['std_output_sigma'] = np.mean(std_output_sigmas[:N])  # Mean std disagreement
        tb_eval_dict['relative_std_output_q'] = np.mean(relative_std_output_qs[:N])  # Relative Q disagreement
        tb_eval_dict['relative_std_output_sigma'] = np.mean(relative_std_output_sigmas[:N])  # Relative std disagreement
        tb_eval_dict['mean_q_bias'] = np.mean(q_bias)  # Mean Q-value bias
        tb_eval_dict['std_q_bias'] = np.std(q_bias)  # Std of Q-value bias
        tb_eval_dict['episode_length'] = len(reward_list)  # Episode length
        tb_eval_dict['overestimation_ratio'] = np.mean(q_bias > 0)  # Overestimation ratio

        import gc
        del obs_list, action_list, log_prob_list, true_q_list, true_qs
        del mean_output_qs, std_output_qs, std_output_sigmas
        gc.collect()

        return tb_eval_dict

    def run_n_episodes(self, n, iteration):
        eval_list = []
        for _ in range(n):
            eval_list.append(self.run_an_episode(iteration, self.render)) 
        avg_tb_eval_dict = {
            k: np.mean([d[k] for d in eval_list]) for k in eval_list[0].keys()
            }
        return avg_tb_eval_dict

    def run_evaluation(self, iteration):
        return self.run_n_episodes(self.num_eval_episode, iteration)