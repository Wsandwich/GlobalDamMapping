import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
output_dir = r'E:\wyj\dam\v4\image\result1'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Data
years = [2010, 2015, 2020]
dam_types = {
    'Embankment_Dam': [4655743, 5712563, 6501762],
    'Gravity_Dam': [15409, 16731, 17738],
    'Arch_Dam': [5809, 5945, 6071],
    'Barrage_Dam': [71493, 77377, 83011]
}

# Title mapping (English only)
title_map = {
    'Embankment_Dam': 'Embankment Dam',
    'Gravity_Dam': 'Gravity Dam',
    'Arch_Dam': 'Arch Dam',
    'Barrage_Dam': 'Barrage Dam'
}

# Color mapping
color_map = {
    'Embankment_Dam': '#AE1A2A',
    'Gravity_Dam': '#D6604D',
    'Arch_Dam': '#F3A781',
    'Barrage_Dam': '#FDDBC7'
}

# Custom y-axis ranges and intervals as requested
y_axis_config = {
    'Embankment_Dam': {'min': 4.0e6, 'max': 7.0e6, 'step': 1.0e6},
    'Gravity_Dam': {'min': 15000, 'max': 18000, 'step': 1000},
    'Arch_Dam': {'min': 5700, 'max': 6300, 'step': 200},
    'Barrage_Dam': {'min': 70000, 'max': 85000, 'step': 5000}
}

# Create a line chart for each dam type
for dam_type, values in dam_types.items():
    plt.figure(figsize=(10, 6))
    
    # Create line chart
    plt.plot(years, values, marker='o', linewidth=6.0, markersize=20, color=color_map[dam_type],
             markeredgecolor='black',    # 黑色轮廓
             markeredgewidth=4)          # 轮廓线宽度)
    
    # Calculate growth rate
    growth_rate = ((values[2] - values[0]) / values[0]) * 100
    
    # Add data labels
    for i, value in enumerate(values):
        # Format value display
        if value >= 1000000:
            label_text = f"{value/1000000:.2f}M"
        elif value >= 1000:
            label_text = f"{value/1000:.1f}K"
        else:
            label_text = f"{value}"
    
    # Set x-axis ticks
    plt.xticks(years, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Set custom y-axis range and ticks based on dam type
    config = y_axis_config[dam_type]
    y_ticks = np.arange(config['min'], config['max'] + config['step'], config['step'])
    plt.yticks(y_ticks)
    plt.ylim(config['min'], config['max'])
    
    # Format y-axis tick labels for Embankment dam (show in millions)
    if dam_type == 'Embankment_Dam':
        ax = plt.gca()
        ax.set_yticklabels([f'{y/1e6:.1f}' for y in y_ticks])
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(4.0)  # 设置 x 轴（底部）线的粗细
    ax.spines['left'].set_linewidth(4.0)  # 设置 y 轴（左侧）线的粗细

    # 设置刻度线的粗细和长度
    ax.tick_params(axis='both',       # 应用于两个坐标轴
                which='major',      # 主刻度线
                length=10,           # 刻度线长度
                width=4.0,          # 刻度线粗细
                direction='out',    # 刻度线向外
                labelsize=0)        # 刻度标签字体大小为0（不显示）

    min_year = min(years)  # 2010
    max_year = max(years)  # 2020
    x_margin = 2  # 设置边距大小（年数），可以根据需要调整

    plt.xlim(min_year - x_margin, max_year + x_margin)  # 例如：2008 到 2022

    # Adjust layout
    plt.tight_layout()
    
    # Save image
    plt.savefig(f"{output_dir}/{dam_type}_trend_v2.png", dpi=500, bbox_inches='tight')
    
    # Close current figure
    plt.close()



print(f"Line charts for four dam types have been saved to {output_dir} folder.")