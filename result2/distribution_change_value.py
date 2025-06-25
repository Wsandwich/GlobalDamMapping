import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import pandas as pd
from matplotlib.gridspec import GridSpec

# 文件路径
shp_path = r'E:\wyj\project\dam\result1\output\BasinATLAS_v10_lev06_with_dam_stats.shp'
output_dir = r'E:\wyj\dam\v4\image\result2\images_v3'
csv_output_dir = os.path.join(output_dir, 'bin_statistics')

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)
os.makedirs(csv_output_dir, exist_ok=True)

# 读取流域Shapefile
print("读取流域Shapefile...")
basins = gpd.read_file(shp_path)

# 定义水坝类型映射
dam_type_mapping = {
    'e': 'embankment',  # 土石坝
    'g': 'gravity',     # 重力坝
    'b': 'Barrage',     # 闸坝
    'a': 'arch'         # 拱坝
}

# 计算变化值
for code, full_name in dam_type_mapping.items():
    change_col = f'change_{code}'
    basins[change_col] = basins[f'2020_{code}'] - basins[f'2010_{code}']
    print(f"计算 {full_name} 水坝变化: {change_col}")

# 计算总变化
change_columns = [f'change_{code}' for code in dam_type_mapping.keys()]
basins['change_total'] = basins[change_columns].sum(axis=1)
print("计算总变化完成")

# 为每种水坝类型定义自定义边界数组
bin_ranges = {
    'change_e': np.array([-100, -50, -20, -5, -1, 0, 1, 5, 20, 50, 100]),  # 土石坝
    'change_g': np.array([-50, -20, -10, -5, -1, 0, 1, 5, 10, 20, 50]),     # 重力坝
    'change_b': np.array([-50, -20, -10, -5, -1, 0, 1, 5, 10, 20, 50]),     # 闸坝
    'change_a': np.array([-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20]),       # 拱坝
    'change_total': np.array([-200, -100, -50, -20, -5, 0, 5, 20, 50, 100, 200])  # 总变化
}

# 创建一个函数，用于生成区间统计数据并导出到CSV
def generate_bin_statistics(data, column, bin_edges, output_csv_path):
    # 计算各个区间内的流域数量
    bin_counts = []
    bin_labels = []
    
    # 添加小于最小边界的区间
    min_edge = bin_edges[0]
    count_below = len(data[data[column] < min_edge])
    if count_below > 0:
        bin_counts.append(count_below)
        bin_labels.append(f"< {min_edge}")
    
    # 计算每个区间的流域数量
    for i in range(len(bin_edges) - 1):
        lower = bin_edges[i]
        upper = bin_edges[i+1]
        
        # 对于0值，单独计算
        if lower == 0 and upper > 0:
            # 0值
            count_zero = len(data[data[column] == 0])
            if count_zero > 0:
                bin_counts.append(count_zero)
                bin_labels.append("= 0")
            
            # (0, upper)区间
            count = len(data[(data[column] > 0) & (data[column] <= upper)])
            if count > 0:
                bin_counts.append(count)
                bin_labels.append(f"(0, {upper}]")
        elif lower < 0 and upper == 0:
            # [lower, 0)区间
            count = len(data[(data[column] >= lower) & (data[column] < 0)])
            if count > 0:
                bin_counts.append(count)
                bin_labels.append(f"[{lower}, 0)")
        else:
            # 一般区间[lower, upper]
            if lower != 0 and upper != 0:  # 跳过已处理的0区间
                count = len(data[(data[column] >= lower) & (data[column] <= upper)])
                if count > 0:
                    bin_counts.append(count)
                    bin_labels.append(f"[{lower}, {upper}]")
    
    # 添加大于最大边界的区间
    max_edge = bin_edges[-1]
    count_above = len(data[data[column] > max_edge])
    if count_above > 0:
        bin_counts.append(count_above)
        bin_labels.append(f"> {max_edge}")
    
    # 创建DataFrame并保存到CSV
    bin_df = pd.DataFrame({
        'Bin_Range': bin_labels,
        'Basin_Count': bin_counts,
        'Percentage': [count / len(data) * 100 for count in bin_counts]
    })
    
    # 添加总计行
    total_count = sum(bin_counts)
    bin_df.loc[len(bin_df)] = ["Total", total_count, 100.0]
    
    # 保存到CSV
    bin_df.to_csv(output_csv_path, index=False)
    print(f"区间统计数据已保存至: {output_csv_path}")
    
    return bin_df

# 创建一个函数，用于绘制单个分布图
def plot_distribution(data, column, full_name, output_path, bin_edges=None):
    # 排除零值以便更好地观察非零变化
    non_zero_data = data[data[column] != 0][column]
    
    if len(non_zero_data) == 0:
        print(f"警告: {full_name} 没有非零变化值")
        return
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. 完整分布图 (包括零值)
    ax1 = fig.add_subplot(gs[0, :])
    
    # 如果提供了bin_edges，使用它们来绘制直方图
    if bin_edges is not None:
        sns.histplot(data[column], bins=bin_edges, kde=True, ax=ax1)
    else:
        sns.histplot(data[column], kde=True, ax=ax1)
    
    ax1.set_title(f'Distribution of {full_name} Dam Change (2020-2010)', fontsize=15)
    ax1.set_xlabel('Change Value', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    
    # 添加统计信息
    stats_text = (
        f"Mean: {data[column].mean():.2f}\n"
        f"Median: {data[column].median():.2f}\n"
        f"Std Dev: {data[column].std():.2f}\n"
        f"Min: {data[column].min()}\n"
        f"Max: {data[column].max()}\n"
        f"Non-zero count: {len(non_zero_data)} ({len(non_zero_data)/len(data)*100:.1f}%)"
    )
    ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, 
             fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # 2. 非零正值分布图
    ax2 = fig.add_subplot(gs[1, 0])
    positive_data = data[data[column] > 0][column]
    if len(positive_data) > 0:
        # 如果提供了bin_edges，过滤出大于0的边界
        if bin_edges is not None:
            positive_bins = bin_edges[bin_edges >= 0]
            if len(positive_bins) > 1:  # 确保至少有两个边界点
                sns.histplot(positive_data, bins=positive_bins, kde=True, color='green', ax=ax2)
            else:
                sns.histplot(positive_data, kde=True, color='green', ax=ax2)
        else:
            sns.histplot(positive_data, kde=True, color='green', ax=ax2)
            
        ax2.set_title(f'Positive Changes', fontsize=12)
        ax2.set_xlabel('Change Value', fontsize=10)
        ax2.text(0.02, 0.95, f"Count: {len(positive_data)}\nMean: {positive_data.mean():.2f}", 
                transform=ax2.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    else:
        ax2.text(0.5, 0.5, "No positive changes", ha='center', va='center', fontsize=12)
    
    # 3. 非零负值分布图
    ax3 = fig.add_subplot(gs[1, 1])
    negative_data = data[data[column] < 0][column]
    if len(negative_data) > 0:
        # 如果提供了bin_edges，过滤出小于0的边界
        if bin_edges is not None:
            negative_bins = bin_edges[bin_edges <= 0]
            if len(negative_bins) > 1:  # 确保至少有两个边界点
                sns.histplot(negative_data, bins=negative_bins, kde=True, color='red', ax=ax3)
            else:
                sns.histplot(negative_data, kde=True, color='red', ax=ax3)
        else:
            sns.histplot(negative_data, kde=True, color='red', ax=ax3)
            
        ax3.set_title(f'Negative Changes', fontsize=12)
        ax3.set_xlabel('Change Value', fontsize=10)
        ax3.text(0.02, 0.95, f"Count: {len(negative_data)}\nMean: {negative_data.mean():.2f}", 
                transform=ax3.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    else:
        ax3.text(0.5, 0.5, "No negative changes", ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图像已保存至: {output_path}")

# 为每种水坝类型和总变化生成区间统计并绘制分布图
for code, full_name in dam_type_mapping.items():
    change_col = f'change_{code}'
    
    # 获取该类型的bin边界
    bin_edges = bin_ranges.get(change_col, None)
    
    # 生成区间统计并导出到CSV
    if bin_edges is not None:
        csv_output_path = os.path.join(csv_output_dir, f'bin_statistics_{full_name}_2020-2010.csv')
        generate_bin_statistics(basins, change_col, bin_edges, csv_output_path)
    
    # 绘制分布图
    output_path = os.path.join(output_dir, f'distribution_{full_name}_2020-2010.png')
    plot_distribution(basins, change_col, full_name, output_path, bin_edges)

# 处理总变化
# 生成区间统计并导出到CSV
bin_edges_total = bin_ranges.get('change_total', None)
if bin_edges_total is not None:
    csv_output_path_total = os.path.join(csv_output_dir, 'bin_statistics_total_2020-2010.csv')
    generate_bin_statistics(basins, 'change_total', bin_edges_total, csv_output_path_total)

# 绘制总变化分布图
output_path_total = os.path.join(output_dir, 'distribution_total_2020-2010.png')
plot_distribution(basins, 'change_total', 'Total', output_path_total, bin_edges_total)

# 另外创建一个对数变换后的分布图，看看转换效果
print("生成对数变换后的分布图...")

def plot_log_transformed_distribution(data, column, full_name, output_path):
    # 创建对数变换的列（保留符号）
    log_col = f'log_{column}'
    data[log_col] = np.sign(data[column]) * np.log1p(np.abs(data[column]))
    
    # 排除零值
    non_zero_data = data[data[column] != 0]
    
    if len(non_zero_data) == 0:
        print(f"警告: {full_name} 没有非零变化值")
        return
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. 原始值与对数变换值对比
    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(non_zero_data[column], non_zero_data[log_col], alpha=0.5)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_title(f'Original vs Log-Transformed Values for {full_name}', fontsize=15)
    ax1.set_xlabel('Original Change Value', fontsize=12)
    ax1.set_ylabel('Log-Transformed Value', fontsize=12)
    
    # 2. 对数变换后的分布
    ax2 = fig.add_subplot(gs[1, :])
    sns.histplot(data[log_col], kde=True, ax=ax2)
    ax2.set_title(f'Distribution of Log-Transformed {full_name} Dam Change', fontsize=15)
    ax2.set_xlabel('Log-Transformed Value', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    
    # 添加统计信息
    stats_text = (
        f"Mean: {data[log_col].mean():.2f}\n"
        f"Median: {data[log_col].median():.2f}\n"
        f"Std Dev: {data[log_col].std():.2f}\n"
        f"Min: {data[log_col].min():.2f}\n"
        f"Max: {data[log_col].max():.2f}"
    )
    ax2.text(0.02, 0.95, stats_text, transform=ax2.transAxes, 
             fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图像已保存至: {output_path}")

# 为每种水坝类型和总变化绘制对数变换分布图
for code, full_name in dam_type_mapping.items():
    change_col = f'change_{code}'
    output_path = os.path.join(output_dir, f'log_distribution_{full_name}_2020-2010.png')
    plot_log_transformed_distribution(basins, change_col, full_name, output_path)

# 绘制总变化的对数变换分布图
output_path_total = os.path.join(output_dir, 'log_distribution_total_2020-2010.png')
plot_log_transformed_distribution(basins, 'change_total', 'Total', output_path_total)

print("所有分布图和区间统计生成完成！")