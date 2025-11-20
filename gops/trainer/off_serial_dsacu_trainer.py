#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Serial trainer for off-policy RL algorithms
#  Update Date: 2021-05-21, Shengbo LI: Format Revise
#  Update Date: 2022-04-14, Jiaxin Gao: decrease parameters copy times
#  Update: 2022-12-05, Wenhan Cao: add annotation

__all__ = ["OffSerialTrainer"]

from cmath import inf
import os
import time

import ray
import torch
from torch.utils.tensorboard import SummaryWriter

from gops.utils.common_utils import ModuleOnDevice
from gops.utils.parallel_task_manager import TaskPool
from gops.utils.tensorboard_setup import add_scalars, tb_tags
from gops.utils.log_data import LogData


class OffSerialDsacuTrainer:
    def __init__(self, alg, sampler, buffer, evaluator, **kwargs):
        self.alg = alg
        self.sampler = sampler
        self.buffer = buffer
        self.per_flag = kwargs["buffer_name"] == "prioritized_replay_buffer"
        self.evaluator = evaluator

        # create center network
        self.networks = self.alg.networks
        self.sampler.networks = self.networks

        # initialize center network
        if kwargs["ini_network_dir"] is not None:
            self.networks.load_state_dict(torch.load(kwargs["ini_network_dir"]))

        self.replay_batch_size = kwargs["replay_batch_size"]
        self.max_iteration = kwargs["max_iteration"]
        self.sample_interval = kwargs.get("sample_interval", 1)
        self.log_save_interval = kwargs["log_save_interval"]
        self.apprfunc_save_interval = kwargs["apprfunc_save_interval"]
        self.eval_interval = kwargs["eval_interval"]
        self.best_tar = -inf
        self.save_folder = kwargs["save_folder"]
        self.iteration = 0
        self.overestimation = 0
        self.update_beta = False
        self.early_stage_flag = True
        self.use_optimistic_behavior_policy = kwargs["lambda_upper"]
        self.mix_mode = kwargs["mix_mode"]

        self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)
        # flush tensorboard at the beginning
        add_scalars(
            {tb_tags["alg_time"]: 0, tb_tags["sampler_time"]: 0}, self.writer, 0
        )
        self.writer.flush()

        # pre sampling
        while self.buffer.size < kwargs["buffer_warm_size"]:
            samples, _ = self.sampler.sample()
            self.buffer.add_batch(samples)
        self.sampler_tb_dict = LogData()

        # create evaluation tasks
        self.evluate_tasks = TaskPool()
        self.last_eval_iteration = 0

        self.use_gpu = kwargs["use_gpu"]
        if self.use_gpu:
            self.networks.cuda()

        self.start_time = time.time()

    def step(self):
        # sampling
        if self.iteration % self.sample_interval == 0:
            with ModuleOnDevice(self.networks, "cpu"):
                if self.use_optimistic_behavior_policy:
                    # Mode 1
                    if self.mix_mode == "continue":
                        mix_ratio = 1 - self.networks.beta.item() / self.networks.beta_init
                        if mix_ratio > 1e-4 and mix_ratio < 1 - 1e-4:
                            if torch.rand(1).item() < mix_ratio:
                                self.sampler.use_target_policy = True
                            else:
                                self.sampler.use_target_policy = False
                        elif mix_ratio <= 1e-4:
                            self.sampler.use_target_policy = False
                        else:
                            self.sampler.use_target_policy = True

                    # Mode 2
                    elif self.mix_mode == "step":
                        if self.networks.beta.item() < 1e-6:
                            self.sampler.use_target_policy = True
                        else:
                            self.sampler.use_target_policy = False

                    elif self.mix_mode == "default":
                        self.sampler.use_target_policy = False

                    else:
                        raise ValueError(f"不支持的mix_mode: {self.mix_mode}，当前仅支持 'continue' 和 'step' 模式")
                else:
                    self.sampler.use_target_policy = True
                sampler_samples, sampler_tb_dict = self.sampler.sample()
            self.buffer.add_batch(sampler_samples)
            self.sampler_tb_dict.add_average(sampler_tb_dict)

        # replay
        replay_samples = self.buffer.sample_batch(self.replay_batch_size)

        # learning
        if self.use_gpu:
            for k, v in replay_samples.items():
                replay_samples[k] = v.cuda()

        self.networks.train()
        if self.per_flag:
            alg_tb_dict, idx, new_priority = self.alg.local_update(
                replay_samples, 
                self.iteration, 
                self.update_beta, 
                self.overestimation
            )
            self.buffer.update_batch(idx, new_priority)
        else:
            alg_tb_dict = self.alg.local_update(
                replay_samples, 
                self.iteration, 
                self.update_beta, 
                self.overestimation
            )
        self.networks.eval()
        self.update_beta = False

        # log
        if self.iteration % self.log_save_interval == 0:
            print("Iter = ", self.iteration)
            add_scalars(alg_tb_dict, self.writer, step=self.iteration)
            add_scalars(self.sampler_tb_dict.pop(), self.writer, step=self.iteration)

        # save
        if self.iteration % self.apprfunc_save_interval == 0:
            self.save_apprfunc()

        # evaluate
        if self.iteration - self.last_eval_iteration >= self.eval_interval:
            if self.evluate_tasks.count == 0:
                # There is no evaluation task, add one.
                self._add_eval_task()
            elif self.evluate_tasks.completed_num == 1:
                # Evaluation tasks is completed, log data and add another one.
                objID = next(self.evluate_tasks.completed())[1]
                avg_tb_eval_dict = ray.get(objID)
                if self.early_stage_flag:
                    if avg_tb_eval_dict['mean_q_bias'] > 0:
                        self.early_stage_flag = False
                    self.overestimation = 0
                else:
                    self.overestimation = avg_tb_eval_dict['mean_q_bias']
                self.update_beta = True
                self._add_eval_task()

                total_avg_return = avg_tb_eval_dict['total_avg_return']
                if (
                    total_avg_return >= self.best_tar
                    and self.iteration >= self.max_iteration / 5
                ):
                    self.best_tar = total_avg_return
                    print("Best return = {}!".format(str(self.best_tar)))

                    for filename in os.listdir(self.save_folder + "/apprfunc/"):
                        if filename.endswith("_opt.pkl"):
                            os.remove(self.save_folder + "/apprfunc/" + filename)

                    torch.save(
                        self.networks.state_dict(),
                        self.save_folder
                        + "/apprfunc/apprfunc_{}_opt.pkl".format(self.iteration),
                    )

                self.writer.add_scalar(
                    tb_tags["Buffer RAM of RL iteration"],
                    self.buffer.__get_RAM__(),
                    self.iteration,
                )
                self.writer.add_scalar(
                    tb_tags["TAR of RL iteration"], 
                    total_avg_return, 
                    self.iteration
                )
                self.writer.add_scalar(
                    tb_tags["TAR of replay samples"],
                    total_avg_return,
                    self.iteration * self.replay_batch_size,
                )
                self.writer.add_scalar(
                    tb_tags["TAR of total time"],
                    total_avg_return,
                    int(time.time() - self.start_time),
                )
                self.writer.add_scalar(
                    tb_tags["TAR of collected samples"],
                    total_avg_return,
                    self.sampler.get_total_sample_number(),
                )
                for k,v in avg_tb_eval_dict.items():
                    if k == 'total_avg_return':
                        continue
                    else:
                        self.writer.add_scalar(tb_tags[k], v, self.iteration)

    def train(self):
        while self.iteration < self.max_iteration:
            self.step()
            self.iteration += 1

        self.save_apprfunc()
        self.writer.flush()

    def save_apprfunc(self):
        torch.save(
            self.networks.state_dict(),
            self.save_folder + "/apprfunc/apprfunc_{}.pkl".format(self.iteration),
        )

    def _add_eval_task(self):
        with ModuleOnDevice(self.networks, "cpu"):
            self.evaluator.load_state_dict.remote(self.networks.state_dict())
        self.evluate_tasks.add(
            self.evaluator,
            self.evaluator.run_evaluation.remote(self.iteration)
        )
        self.last_eval_iteration = self.iteration
