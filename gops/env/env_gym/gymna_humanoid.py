#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Bench Hard Environment


import gymnasium as gym


def env_creator(**kwargs):
    try:
        terminate_when_unhealthy = kwargs.get("terminate_when_unhealthy", True)
        return gym.make("Humanoid-v3", render_mode="rgb_array", terminate_when_unhealthy=terminate_when_unhealthy)
    except:
        raise ModuleNotFoundError(
            "Warning:  mujoco, mujoco-py and MSVC are not installed properly"
        )


if __name__ == "__main__":
    from gops.env.wrapper.tensor import TensorWrapper
    import torch
    env = env_creator(**{"tensor_env":True})
    env = TensorWrapper(env)
    obs, info = env.reset(seed=1)
    env.action_space.seed(seed=1)
    print(obs[:10])
    for i in range(100):
        a = env.action_space.sample()
        a = torch.from_numpy(a)
        if i == 1:
            print(a)
            print(frame)
        out = env.step(a)
        frame = env.render()