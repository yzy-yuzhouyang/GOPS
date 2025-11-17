import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tensorboard.backend.event_processing import event_accumulator

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Visualize TensorBoard event files')
    parser.add_argument('--logdir', type=str, required=True,
                        help='Directory containing TensorBoard event files')
    parser.add_argument('--tags', type=str, nargs=3, 
                        default=['0_dsact_std', '0_std1', '0_std2'],
                        help='Three tags to visualize (default: loss accuracy learning_rate)')
    parser.add_argument('--smoothing', type=float, default=0.0,
                        help='Smoothing factor for curves (0.0 to 1.0)')
    parser.add_argument('--y-label', type=str, default='Value',
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

def extract_data(event_path, tags):
    """从event文件中提取指定tag的数据"""
    data = {tag: {'steps': [], 'values': []} for tag in tags}
    
    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()
    for name in tags:
        curve = ea.Scalars("Track/" + name)
        data[name]['steps'] = [s.step / 1e6 for s in curve]
        data[name]['values'] = [s.value for s in curve]
    
    return data

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

def generate_filename(log_dir, tags):
    """生成基于tags的文件名"""
    # 将tags连接为文件名
    tag_str = "_".join(tags)
    # 移除可能引起文件问题的字符
    safe_tag_str = "".join(c for c in tag_str if c.isalnum() or c in ('_', '-')).rstrip()
    filename = f"tb_plot_{safe_tag_str}.png"
    save_dir = os.path.join(log_dir, "tb_figs")
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, filename)

def main():
    # 解析参数
    args = parse_arguments()
    
    # 查找event文件
    event_file = find_event_file(args.logdir)
    print(f"Using event file: {event_file}")
    
    # 提取数据
    data = extract_data(event_file, args.tags)
    
    # 创建单个图表
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle('TensorBoard Metrics Visualization', fontsize=16, fontweight='bold')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    # 绘制所有曲线在同一图表上
    for idx, tag in enumerate(args.tags):
        if not data[tag]['steps']:
            print(f"Warning: No data found for tag '{tag}'")
            continue
            
        steps = data[tag]['steps']
        values = data[tag]['values']
        
        # 应用平滑
        if args.smoothing > 0:
            values = smooth_data(values, args.smoothing)
        
        # 绘制曲线
        ax.plot(steps, values, color=colors[idx], linewidth=2.5, alpha=0.8, label=tag)
    
    # 设置图表属性
    ax.set_xlabel('Step (Million)', fontsize=12)
    ax.set_ylabel(args.y_label, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 添加图例
    ax.legend(loc='best', fontsize=12)
    
    # 添加统计信息到图表
    stats_text = ""
    for idx, tag in enumerate(args.tags):
        if data[tag]['values']:
            values = data[tag]['values']
            if args.smoothing > 0:
                values = smooth_data(values, args.smoothing)
            mean_val = np.mean(values)
            min_val = np.min(values)
            max_val = np.max(values)
            stats_text += f'{tag}: Mean={mean_val:.4f}, Min={min_val:.4f}, Max={max_val:.4f}\n'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # 生成并保存图像文件
    output_path = generate_filename(args.logdir, args.tags)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_path}")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    main()