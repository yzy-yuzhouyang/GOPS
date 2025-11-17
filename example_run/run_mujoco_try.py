#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: run a closed-loop system
#  Update: 2022-12-05, Congsheng Zhang: create file


from gops.sys_simulator.sys_run2 import PolicyRunner

import os

import os
from pyvirtualdisplay import Display

# 启动虚拟显示
display = Display(visible=0, size=(1400, 900))
display.start()

os.environ['MUJOCO_GL'] = 'glfw'  # osmesa glfw egl
print("MUJOCO_GL is set to:", os.environ.get('MUJOCO_GL'))

runner = PolicyRunner(
    log_policy_dir_list=["../results/gym_humanoid/DSACU3SCALE-2-0.02-False-6e-05-12345_251017-133523"],
    trained_policy_iteration_list=["1432500_opt"],
    legend_list=["DSACU"],
    use_opt=False,  # Use optimal solution for comparison
    save_render=False,
    render_safety_gym=True,
)

runner.run(1)
