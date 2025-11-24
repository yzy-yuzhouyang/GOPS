#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Bench Hard Environment

from gops.env.env_dmc.dmcontrol import make_env


def env_creator(**kwargs):
    try:
        return make_env('dog-run')
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
    print(obs, info)
    for i in range(100):
        a = env.action_space.sample()
        a = torch.from_numpy(a)
        if i == 0:
            print(a)
        out = env.step(a)