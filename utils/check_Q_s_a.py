import numpy as np
import os

# 指定包含npy文件的目录路径
directory_path = "/home/yuzhouyang/code/GOPS/results/gym_ant/DSACT-1234_250906-210219/evaluator/track"  # 请替换为实际的目录路径

# 获取目录中的所有npy文件
npy_files = [f for f in os.listdir(directory_path) if f.endswith('.npy')]

for file_name in npy_files:
    file_path = os.path.join(directory_path, file_name)
    print(f"\n处理文件: {file_name}")
    
    try:
        data = np.load(file_path, allow_pickle=True).item()
        
        if "Q" in data:
            q_values = data["Q"]
            print(f"Q值 (前10个): {q_values[:10] if len(q_values) > 10 else q_values}")
            print(f"Q值范围: {np.min(q_values):.4f} 到 {np.max(q_values):.4f}")
            print(f"平均Q值: {np.mean(q_values):.4f}")
        else:
            print("文件中未找到'Q'键")
            
    except Exception as e:
        print(f"处理文件时出错: {e}")