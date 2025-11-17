import os
import shutil
import re

def copy_selected_files(source_base_folder, target_base_folder):
    """
    复制指定基础文件夹中的所有子文件夹内容到新文件夹，按照特定规则筛选文件
    
    Args:
        source_base_folder: 源基础文件夹路径（如gym_ant）
        target_base_folder: 目标基础文件夹路径
    """
    
    # 检查源基础文件夹是否存在
    if not os.path.exists(source_base_folder):
        print(f"错误: 源基础文件夹 {source_base_folder} 不存在!")
        return
    
    # 获取源基础文件夹下的所有子文件夹
    subfolders = [f for f in os.listdir(source_base_folder) 
                  if os.path.isdir(os.path.join(source_base_folder, f))]
    
    if not subfolders:
        print(f"警告: {source_base_folder} 中没有子文件夹，跳过")
        return
    
    print(f"在 {source_base_folder} 中找到 {len(subfolders)} 个子文件夹")
    
    for subfolder in subfolders:
        source_folder = os.path.join(source_base_folder, subfolder)
        target_folder = os.path.join(target_base_folder, subfolder)
        
        # 确保目标文件夹存在
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        print(f"处理子文件夹: {source_folder} -> {target_folder}")
        
        # 需要排除的文件夹
        exclude_dirs = {'figure', 'evaluator'}
        
        # 遍历源子文件夹
        for root, dirs, files in os.walk(source_folder):
            # 获取相对路径
            relative_path = os.path.relpath(root, source_folder)
            
            # 跳过排除的文件夹
            if any(excluded in relative_path.split(os.sep) for excluded in exclude_dirs):
                continue
            
            # 创建对应的目标文件夹
            target_path = os.path.join(target_folder, relative_path)
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            
            # 处理appfunc文件夹中的文件
            if 'apprfunc' in root or 'apprfunc' in relative_path:
                process_appfunc_files(root, files, target_path)
            else:
                # 复制其他文件夹中的所有文件（除了排除的文件夹）
                for file in files:
                    source_file = os.path.join(root, file)
                    target_file = os.path.join(target_path, file)
                    shutil.copy2(source_file, target_file)
                    print(f"复制: {source_file} -> {target_file}")

def process_appfunc_files(source_dir, files, target_dir):
    """
    处理appfunc文件夹中的文件，按照特定规则筛选
    
    Args:
        source_dir: 源目录
        files: 文件列表
        target_dir: 目标目录
    """
    
    # 存储所有数字文件的信息
    number_files = []
    opt_files = []
    
    # 正则表达式匹配文件名中的数字
    pattern = r'apprfunc_(\d+)\.pkl'
    
    for file in files:
        source_file = os.path.join(source_dir, file)
        
        # 检查是否包含opt后缀
        if '_opt.pkl' in file:
            opt_files.append((file, source_file))
            print(f"保留opt文件: {file}")
            continue
        
        # 提取数字
        match = re.search(pattern, file)
        if match:
            number = int(match.group(1))
            number_files.append((number, file, source_file))
    
    # 复制所有opt文件
    for opt_file, source_path in opt_files:
        target_file = os.path.join(target_dir, opt_file)
        shutil.copy2(source_path, target_file)
        print(f"复制opt文件: {source_path} -> {target_file}")
    
    if number_files:
        # 按数字排序
        number_files.sort(key=lambda x: x[0])
        
        # 保留数字最大的文件
        max_number, max_file, max_source = number_files[-1]
        target_file = os.path.join(target_dir, max_file)
        shutil.copy2(max_source, target_file)
        print(f"复制最大数字文件: {max_source} -> {target_file}")
        
        # 检查是否有1500000的文件
        for number, file, source_path in number_files:
            if number == 1500000:
                target_file = os.path.join(target_dir, file)
                shutil.copy2(source_path, target_file)
                print(f"复制1500000文件: {source_path} -> {target_file}")
                break

def main():
    # 硬编码源基础文件夹（请根据实际情况修改）
    source_base_folders = [
        # "results/gym_ant",
        "results/gym_humanoid"
        # 可以添加更多基础文件夹
    ]
    
    print("开始复制文件...")
    print(f"源基础文件夹: {source_base_folders}")
    
    target_prefix = "filtered_"
    try:
        for source_base_folder in source_base_folders:
            target_base_folder = target_prefix + source_base_folder
            if not os.path.exists(target_base_folder):
                os.makedirs(target_base_folder)
            copy_selected_files(source_base_folder, target_base_folder)
        print("文件复制完成！")
    except Exception as e:
        print(f"复制过程中出现错误: {e}")

if __name__ == "__main__":
    main()