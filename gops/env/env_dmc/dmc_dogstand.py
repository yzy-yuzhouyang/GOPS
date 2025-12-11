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
        return make_env('dog-stand')
    except:
        raise ModuleNotFoundError(
            "Warning:  mujoco, mujoco-py and MSVC are not installed properly"
        )


if __name__ == "__main__":
    env = env_creator()
    env.reset()
    for i in range(100):
        a = env.action_space.sample()
        out = env.step(a)
        # env.render()