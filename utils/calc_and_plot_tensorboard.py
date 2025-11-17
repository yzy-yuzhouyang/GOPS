import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tensorboard.backend.event_processing import event_accumulator

'''
Example:
    python GOPS/utils/calc_and_plot_tensorboard.py \
    --logdir="/home/yuzhouyang/code/GOPS/results/gym_humanoid/DSACTMinus-0.05-0.5-False-1234_250915-112114" \
    --numerator-tag="DSAC2/critic_avg_std1-RL iter" \
    --denominator-tag="DSAC2/critic_avg_q1-RL iter"
'''

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Visualize ratio of two TensorBoard metrics')
    parser.add_argument('--logdir', type=str, required=True,
                        help='Directory containing TensorBoard event files')
    parser.add_argument('--numerator-tag', type=str, required=True,
                        help='Tag for numerator (to be divided)')
    parser.add_argument('--denominator-tag', type=str, required=True,
                        help='Tag for denominator (divisor)')
    parser.add_argument('--ratio-name', type=str, default='Ratio',
                        help='Name for the ratio (used in legend and labels)')
    parser.add_argument('--smoothing', type=float, default=0.0,
                        help='Smoothing factor for curves (0.0 to 1.0)')
    parser.add_argument('--y-label', type=str, default='Ratio Value',
                        help='Label for Y-axis')
    return parser.parse_args()

def smooth_data(data, weight=0.9):
    """应用指数平滑到数据"""
    smoothed = []
    last = data[0]
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def extract_and_align_data(event_path, numerator_tag, denominator_tag):
    """从event文件中提取并对齐两个tag的数据"""
    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()
    
    # 提取分子数据
    numerator_curve = ea.Scalars(numerator_tag)
    numerator_steps = [s.step for s in numerator_curve]
    numerator_values = [s.value for s in numerator_curve]
    
    # 提取分母数据
    denominator_curve = ea.Scalars(denominator_tag)
    denominator_steps = [s.step for s in denominator_curve]
    denominator_values = [s.value for s in denominator_curve]
    
    # 对齐数据 - 找到共同的step点
    common_steps = sorted(set(numerator_steps) & set(denominator_steps))
    
    # 获取共同step对应的值
    aligned_numerator = []
    aligned_denominator = []
    
    # 创建步数到值的映射
    numerator_dict = dict(zip(numerator_steps, numerator_values))
    denominator_dict = dict(zip(denominator_steps, denominator_values))
    
    for step in common_steps:
        aligned_numerator.append(numerator_dict[step])
        aligned_denominator.append(denominator_dict[step])
    
    # 计算比率（处理除零情况）
    ratio_values = []
    for num, den in zip(aligned_numerator, aligned_denominator):
        if den == 0:
            ratio_values.append(np.nan)  # 避免除零错误
        else:
            result = abs(num / den)
            result = min(result, 0.2)
            ratio_values.append(result)
    
    # 将步数转换为百万单位
    common_steps = [step / 1e6 for step in common_steps]
    
    return {
        'steps': common_steps,
        'numerator': aligned_numerator,
        'denominator': aligned_denominator,
        'ratio': ratio_values
    }

def find_event_file(log_dir):
    """在日志目录中查找最新的event文件"""
    event_files = []
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, file))
    
    if not event_files:
        raise FileNotFoundError(f"No event files found in {log_dir}")
    
    # 返回最新的event文件
    return max(event_files, key=os.path.getmtime)

def generate_filename(log_dir, numerator_tag, denominator_tag):
    """生成基于tags的文件名"""
    # 将tags连接为文件名
    tag_str = f"{numerator_tag}_over_{denominator_tag}"
    # 移除可能引起文件问题的字符
    safe_tag_str = "".join(c for c in tag_str if c.isalnum() or c in ('_', '-')).rstrip()
    filename = f"ratio_plot_{safe_tag_str}.png"
    save_dir = os.path.join(log_dir, "tb_figs")
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, filename)

def main():
    # 解析参数
    args = parse_arguments()
    
    # 查找event文件
    event_file = find_event_file(args.logdir)
    print(f"Using event file: {event_file}")
    
    # 提取并对齐数据
    data = extract_and_align_data(event_file, args.numerator_tag, args.denominator_tag)
    
    if not data['steps']:
        print("Error: No common steps found between the two tags!")
        return
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(f'{args.numerator_tag} / {args.denominator_tag}', fontsize=16, fontweight='bold')
    
    # 应用平滑
    ratio_values = data['ratio']
    if args.smoothing > 0:
        # 先处理NaN值
        valid_indices = [i for i, v in enumerate(ratio_values) if not np.isnan(v)]
        valid_values = [ratio_values[i] for i in valid_indices]
        valid_steps = [data['steps'][i] for i in valid_indices]
        
        if valid_values:
            smoothed_values = smooth_data(valid_values, args.smoothing)
            
            # 绘制平滑后的比率曲线
            ax.plot(valid_steps, smoothed_values, color='#FF6B6B', linewidth=2.5, 
                   alpha=0.8, label=f'{args.ratio_name} (smoothed)')
    
    # 绘制原始比率曲线
    ax.plot(data['steps'], ratio_values, color='#4ECDC4', linewidth=1.5, 
           alpha=0.6, label=args.ratio_name)
    
    # 设置图表属性
    ax.set_xlabel('Step (Million)', fontsize=12)
    ax.set_ylabel(args.y_label, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 添加图例
    ax.legend(loc='best', fontsize=12)
    
    # 添加统计信息到图表
    valid_ratios = [r for r in ratio_values if not np.isnan(r)]
    if valid_ratios:
        mean_val = np.mean(valid_ratios)
        min_val = np.min(valid_ratios)
        max_val = np.max(valid_ratios)
        stats_text = f'Mean={mean_val:.4f}\nMin={min_val:.4f}\nMax={max_val:.4f}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # 生成并保存图像文件
    output_path = generate_filename(args.logdir, args.numerator_tag, args.denominator_tag)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_path}")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    main()