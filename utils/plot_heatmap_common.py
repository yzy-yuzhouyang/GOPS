import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse
from collections import defaultdict
from matplotlib import rcParams
import matplotlib.colors as colors

# 设置中文字体支持（如果需要）
# rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_data(file_path):
    """
    从.npy文件中加载数据，并计算next_std
    返回包含所有数据的字典
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        # 检查数据格式
        if isinstance(data, np.ndarray):
            data = data.item()  # 转换为字典
        
        # 确保数据中包含所需的列表
        if 'std' not in data or 'bias' not in data or 'Q' not in data:
            print(f"文件 {file_path} 中未找到 'std', 'Q' 或 'bias'")
            return None
        
        # 准备返回的数据字典
        result = {
            'iter': data.get("iter"),
            'std': data.get('std'),
            'Q': data.get('Q'),
            'bias': data.get('bias'),
            'next_std': data.get('std', [])[1:],  # std向后平移一位
            'diff': data.get('diff'),
            'std_rel_diff': data.get('std_rel_diff'),
            'rel_diff': data.get('rel_diff')
        }
        
        return result
        
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return None

def plot_heatmap(x_data, y_data, c_data, 
                 x_label="X", y_label="Y", c_label="Color",
                 title="Heatmap", save_path=None):
    """
    通用热图绘制函数
    
    参数:
    x_data: 横坐标数据
    y_data: 纵坐标数据
    c_data: 颜色数据
    x_label: 横坐标标签
    y_label: 纵坐标标签
    c_label: 颜色标签
    title: 图表标题
    save_path: 保存路径
    """
    if x_data is None or y_data is None or c_data is None or len(x_data) == 0:
        print("没有有效数据可供绘制")
        return
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 创建二维直方图（热图）
    hb = plt.hexbin(x_data, y_data, C=c_data, 
                   gridsize=50, cmap='viridis', 
                   reduce_C_function=np.mean,
                   mincnt=1)  # 只显示至少有1个点的格子
    
    # 添加颜色条
    cb = plt.colorbar(hb, label=c_label)
    
    # 添加标题和标签
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示图形
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"图形已保存至: {save_path}")
    
    plt.show()

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='绘制Q、Std、Bias、Next_Std和Diff的热图')
    parser.add_argument('--x', type=str, default='Q', 
                       choices=['Q', 'std', 'bias', 'next_std', 'diff', 'std_rel_diff', 'rel_diff'],
                       help='横坐标变量 (Q, std, bias, next_std, diff, std_rel_diff, rel_diff)')
    parser.add_argument('--y', type=str, default='std', 
                       choices=['Q', 'std', 'bias', 'next_std', 'diff', 'std_rel_diff', 'rel_diff'],
                       help='纵坐标变量 (Q, std, bias, next_std, diff, std_rel_diff, rel_diff)')
    parser.add_argument('--c', type=str, default='bias', 
                       choices=['Q', 'std', 'bias', 'next_std', 'diff', 'std_rel_diff', 'rel_diff'],
                       help='颜色变量 (Q, std, bias, next_std, diff, std_rel_diff, rel_diff)')
    parser.add_argument('--exp', type=str, 
                       default='/home/yuzhouyang/code/GOPS/results/gym_ant/DSACTMinus-0.1-0.7-1234_250905-161252',
                       help='实验文件夹路径')
    parser.add_argument('--iterations', type=int, nargs='+',
                       default=[10000, 20000, 25000, 30000, 35000, 40000, 45000, 50000],
                       help='要分析的迭代次数列表')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_arguments()
    
    # 验证参数
    variables = set([args.x, args.y, args.c])
    if len(variables) != 3:
        print("错误：x_var、y_var和c_var必须是三个不同的变量")
        return
    
    # 创建标签映射
    label_map = {
        'Q': 'Q Value',
        'std': 'Std Output',
        'bias': 'Bias',
        'next_std': 'Next Std Output',
        'diff': 'Difference',
        'std_rel_diff': 'Std Relative Difference',
        'rel_diff': 'Relative Difference'
    }
    
    # 设置数据文件路径
    exp_folder = args.exp
    data_folder = os.path.join(exp_folder, "evaluator/visualize")
    save_folder = os.path.join(exp_folder, f"figs/visualize/{args.x}_{args.y}_{args.c}")
    os.makedirs(save_folder, exist_ok=True)
    
    # 查找数据文件
    file_pattern = "*.npy"
    file_paths = glob.glob(os.path.join(data_folder, file_pattern))
    
    if not file_paths:
        print(f"在 {data_folder} 中未找到 {file_pattern} 文件")
        return
    
    # 初始化数据字典
    data_dict = defaultdict(lambda: defaultdict(list))
    target_iter = set(args.iterations)
    
    # 加载数据
    for file_path in file_paths:
        print(f"正在处理文件: {os.path.basename(file_path)}")
        data = load_data(file_path)
        if data is None:
            continue
            
        iter = data['iter']
        if iter % 100000 == 0 and iter not in target_iter:
            target_iter.add(iter)
            
        if iter in target_iter:
            # 获取有效数据（移除最后一个元素以匹配next_std长度）
            valid_std = data['std'][:-1]
            valid_q = data['Q'][:-1]
            valid_bias = data['bias'][:-1]
            valid_next_std = data['next_std']
            
            # 添加到数据字典
            data_dict[iter]["std"].extend(valid_std)
            data_dict[iter]["Q"].extend(valid_q)
            data_dict[iter]["bias"].extend(valid_bias)
            data_dict[iter]["next_std"].extend(valid_next_std)
            
            # 添加可选数据
            for key in ['diff', 'std_rel_diff', 'rel_diff']:
                if data[key] is not None:
                    valid_data = data[key][:-1] if len(data[key]) > len(valid_std) else data[key]
                    data_dict[iter][key].extend(valid_data)
    
    # 对迭代次数进行排序
    target_iter = sorted(target_iter)
    
    # 为每个迭代绘制热图
    for iter in target_iter:
        if iter not in data_dict:
            print(f"Iter {iter} is not in records. ")
            continue
            
        # 转换为numpy数组以便处理
        all_std = np.array(data_dict[iter]["std"])
        all_q = np.array(data_dict[iter]["Q"])
        all_bias = np.array(data_dict[iter]["bias"])
        all_next_std = np.array(data_dict[iter]["next_std"])
        
        # 获取可选数据
        optional_data = {}
        for key in ['diff', 'std_rel_diff', 'rel_diff']:
            if key in data_dict[iter]:
                optional_data[key] = np.array(data_dict[iter][key])
            else:
                optional_data[key] = None
        
        # 确保数据长度一致
        min_length = len(all_std)
        all_q = all_q[:min_length]
        all_bias = all_bias[:min_length]
        all_next_std = all_next_std[:min_length]
        
        for key in optional_data:
            if optional_data[key] is not None:
                optional_data[key] = optional_data[key][:min_length]
        
        # 打印基本统计信息
        print(f"\n迭代 {iter} 的数据统计:")
        print(f"数据点数: {len(all_std)}")
        print(f"Q 范围: [{all_q.min():.4f}, {all_q.max():.4f}]")
        print(f"Std 范围: [{all_std.min():.4f}, {all_std.max():.4f}]")
        print(f"Bias 范围: [{all_bias.min():.4f}, {all_bias.max():.4f}]")
        print(f"Next_Std 范围: [{all_next_std.min():.4f}, {all_next_std.max():.4f}]")
        
        for key, data in optional_data.items():
            if data is not None:
                print(f"{key} 范围: [{data.min():.4f}, {data.max():.4f}]")
        
        print(f"Q 均值: {all_q.mean():.4f}, 标准差: {all_q.std():.4f}")
        print(f"Std 均值: {all_std.mean():.4f}, 标准差: {all_std.std():.4f}")
        print(f"Bias 均值: {all_bias.mean():.4f}, 标准差: {all_bias.std():.4f}")
        print(f"Next_Std 均值: {all_next_std.mean():.4f}, 标准差: {all_next_std.std():.4f}")
        
        for key, data in optional_data.items():
            if data is not None:
                print(f"{key} 均值: {data.mean():.4f}, 标准差: {data.std():.4f}")
        
        # 根据参数选择数据
        data_map = {
            'Q': all_q,
            'std': all_std,
            'bias': all_bias,
            'next_std': all_next_std,
            'diff': optional_data['diff'] if optional_data['diff'] is not None else np.zeros_like(all_std),
            'std_rel_diff': optional_data['std_rel_diff'] if optional_data['std_rel_diff'] is not None else np.zeros_like(all_std),
            'rel_diff': optional_data['rel_diff'] if optional_data['rel_diff'] is not None else np.zeros_like(all_std)
        }
        
        x_data = data_map[args.x]
        y_data = data_map[args.y]
        c_data = data_map[args.c]
        
        # 检查是否有无效数据
        if (args.x in ['diff', 'std_rel_diff', 'rel_diff'] and 
            optional_data[args.x] is None):
            print(f"错误: 迭代 {iter} 中没有 {args.x} 数据，但被选为 x 轴变量")
            continue
        if (args.y in ['diff', 'std_rel_diff', 'rel_diff'] and 
            optional_data[args.y] is None):
            print(f"错误: 迭代 {iter} 中没有 {args.y} 数据，但被选为 y 轴变量")
            continue
        if (args.c in ['diff', 'std_rel_diff', 'rel_diff'] and 
            optional_data[args.c] is None):
            print(f"错误: 迭代 {iter} 中没有 {args.c} 数据，但被选为颜色变量")
            continue
        
        # 绘制热图
        save_path = os.path.join(save_folder, f"heatmap_{iter}.png")
        title = f"{label_map[args.x]} vs {label_map[args.y]} ({label_map[args.c]}) - Iter {iter}"
        
        plot_heatmap(x_data, y_data, c_data,
                    x_label=label_map[args.x],
                    y_label=label_map[args.y],
                    c_label=label_map[args.c],
                    title=title,
                    save_path=save_path)

if __name__ == "__main__":
    main()