import geopandas as gpd
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm, to_rgba
from matplotlib.cm import ScalarMappable, get_cmap
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib as mpl

# **绘图函数**
def plot_map(data, column, ax, title, output_path, gdf_policies_grouped=None, vmin=None, vmax=None, custom_bins=None):
    
    # 如果没有指定vmin和vmax，根据数据自动计算合适的范围
    if vmin is None or vmax is None and custom_bins is None:
        # 计算有变化数据的绝对值最大值
        max_abs_val = abs(data[data[column] != 0][column]).max()
        if max_abs_val > 0:
            vmin = -max_abs_val
            vmax = max_abs_val
        else:
            vmin = -1
            vmax = 1

    # **分离无变化和有变化的流域**
    data_no_change = data[data[column] == 0]
    data_change = data[data[column] != 0]
    
    # **添加全球海岸线**
    ax.coastlines(resolution='50m', linewidth=0.3, color="black")
    
    # **绘制有变化的流域**
    if not data_change.empty:
        # 设置自定义的颜色映射
        if custom_bins is not None:
            # 创建自定义的边界规范和颜色
            cmap = get_cmap('RdBu_r')
            
            # 处理特殊情况：如果有两个连续的0，将其中一个稍微调整，以创建一个很小的区间
            adjusted_bins = np.array(custom_bins, dtype=float)  # 创建副本以避免修改原始数组
            
            # 检查是否有连续的0
            zero_indices = np.where(adjusted_bins == 0)[0]
            if len(zero_indices) >= 2 and zero_indices[1] - zero_indices[0] == 1:
                # 将第二个0调整为一个很小的正数，以创建一个小区间表示0
                adjusted_bins[zero_indices[1]] = 1e-10
            
            norm = BoundaryNorm(adjusted_bins, cmap.N)
            
            # 绘制有变化的流域，使用自定义的边界规范
            data_change.plot(column=column, ax=ax, cmap=cmap, norm=norm,
                         edgecolor='none', transform=ccrs.PlateCarree())
        else:
            # 使用常规的vmin和vmax
            data_change.plot(column=column, ax=ax, cmap='RdBu_r',
                         edgecolor='none', vmin=vmin, vmax=vmax,
                         transform=ccrs.PlateCarree())
    
    # **绘制无变化的流域（淡灰色填充）- 始终独立处理零值**
    if not data_no_change.empty:
        data_no_change.plot(ax=ax, color='whitesmoke', edgecolor='none',
                        transform=ccrs.PlateCarree())
    
    # **绘制政策线**
    if gdf_policies_grouped is not None:
        for policy_type, color in policy_types.items():
            policy_data = gdf_policies_grouped[gdf_policies_grouped['政策类型'] == policy_type]
            if not policy_data.empty:
                policy_data.plot(
                    ax=ax, edgecolor=color, linewidth=1.5, facecolor='none',
                    transform=ccrs.PlateCarree()
                )
    
    # **设置全球范围**
    ax.set_global()
    # ax.gridlines(draw_labels=False, linewidth=0.2, color='gray', alpha=0.5)
    
    # **去除图框**
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # **设置标题**
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # **添加自定义图例**
    # 颜色渐变条
    cax = plt.axes([0.92, 0.25, 0.02, 0.5])  # 位置[左, 下, 宽, 高]
    
    if custom_bins is not None:
        # 为自定义边界创建颜色条
        cmap = get_cmap('RdBu_r')
        
        # 使用自定义的颜色带来处理不对称边界和零值
        
        # 1. 找到零值位置和各区间所占比例
        zero_indices = np.where(custom_bins == 0)[0]
        
        # 为颜色带创建定制刻度和标签
        tick_locs = []
        tick_labels = []
        
        # 2. 为颜色带创建一个更详细的配色方案，确保每个区间有合适的颜色
        if len(zero_indices) >= 2 and zero_indices[1] - zero_indices[0] == 1:
            # 有专门的0区间
            
            # 计算负值和正值部分的区间数量
            neg_bins_count = zero_indices[0]
            pos_bins_count = len(custom_bins) - zero_indices[1] - 1
            
            # 创建颜色映射
            colors = []
            
            # 蓝色部分 - 负值
            if neg_bins_count > 0:
                blues = plt.cm.Blues_r(np.linspace(0.2, 0.9, neg_bins_count))
                colors.extend(blues)
            
            # 白色 - 0值
            colors.append((1.0, 1.0, 1.0, 1.0))  # 白色表示0
            
            # 红色部分 - 正值
            if pos_bins_count > 0:
                reds = plt.cm.Reds(np.linspace(0.2, 0.9, pos_bins_count))
                colors.extend(reds)
            
            # 创建自定义颜色映射
            custom_cmap = mcolors.ListedColormap(colors)
            norm = BoundaryNorm(custom_bins, custom_cmap.N)
            
            # 创建刻度点和标签
            # 负值部分
            for i in range(neg_bins_count):
                mid_point = (custom_bins[i] + custom_bins[i+1])/2
                tick_locs.append(mid_point)
                tick_labels.append(f"{custom_bins[i]:.1f} to {custom_bins[i+1]:.1f}")
            
            # 0值部分 - 专门处理
            tick_locs.append(0)
            tick_labels.append("= 0")
            
            # 正值部分
            for i in range(zero_indices[1], len(custom_bins)-1):
                mid_point = (custom_bins[i] + custom_bins[i+1])/2
                tick_locs.append(mid_point)
                tick_labels.append(f"{custom_bins[i]:.1f} to {custom_bins[i+1]:.1f}")
        else:
            # 没有专门的0区间，使用常规的RdBu_r
            custom_cmap = cmap
            norm = BoundaryNorm(custom_bins, custom_cmap.N)
            
            # 创建刻度点和标签
            for i in range(len(custom_bins)-1):
                mid_point = (custom_bins[i] + custom_bins[i+1])/2
                tick_locs.append(mid_point)
                tick_labels.append(f"{custom_bins[i]:.1f} to {custom_bins[i+1]:.1f}")
        
        # 创建颜色条
        sm = ScalarMappable(norm=norm, cmap=custom_cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        
        # 设置刻度和标签
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(tick_labels)
    else:
        # 常规颜色条
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(norm=norm, cmap='RdBu_r')
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
    
    cbar.set_label(f'Dam Change ({column})', fontsize=10)
    
    # 添加无变化和政策类型图例
    legend_elements = [mpatches.Patch(facecolor='whitesmoke', edgecolor='none', label='No Change (= 0)')]
    
    if gdf_policies_grouped is not None:
        for policy_type, color in policy_types.items():
            legend_elements.append(
                mpatches.Patch(facecolor='none', edgecolor=color, linewidth=1.5, label=policy_type)
            )
    
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10, framealpha=0.7)
    
    # **保存图像**
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"图像已保存至: {output_path}")
    plt.close()

# **文件路径**
shp_path = r'E:\wyj\project\dam\result1\output\BasinATLAS_v10_lev06_with_dam_stats.shp'
policy_folder = r'E:\wyj\project\dam\draw\数据\政策'
output_base_path = r'E:\wyj\dam\v4\image\result2\images_v7_2015_2020'
csv_output_dir = os.path.join(output_base_path, 'bin_statistics')

# **确保输出目录存在**
os.makedirs(output_base_path, exist_ok=True)
os.makedirs(csv_output_dir, exist_ok=True)

# **读取流域 Shapefile**
print("读取流域 Shapefile...")
basins = gpd.read_file(shp_path)

# **定义水坝类型映射（原始代码中的类型到您的数据中的类型）**
dam_type_mapping = {
    'e': 'embankment',  # 土石坝
    'g': 'gravity',     # 重力坝
    'b': 'Barrage',     # 闸坝
    'a': 'arch'         # 拱坝
}

# **计算四种水坝类型的变化（2020 - 2015）**
for code, full_name in dam_type_mapping.items():
    change_col = f'change_{code}'
    basins[change_col] = basins[f'2020_{code}'] - basins[f'2015_{code}']
    print(f"计算 {full_name} 水坝变化: {change_col}")

# **计算总变化**
change_columns = [f'change_{code}' for code in dam_type_mapping.keys()]
basins['change_total'] = basins[change_columns].sum(axis=1)
print("计算总变化完成")

# **确保坐标系为 WGS84（EPSG:4326）**
if basins.crs is None or basins.crs.to_string() != 'EPSG:4326':
    print("转换坐标系到 WGS84...")
    basins = basins.set_crs(epsg=4326)

# **读取政策文件**
policy_types = {"拆除": "purple", "建设": "green"}
policy_list = []

print("尝试加载政策文件...")
try:
    for policy_type in policy_types.keys():
        policy_dir = os.path.join(policy_folder, policy_type)
        if not os.path.exists(policy_dir):
            print(f"警告: 目录 {policy_dir} 不存在")
            continue
            
        for file in os.listdir(policy_dir):
            if file.endswith("lines.shp"):
                shp_file = os.path.join(policy_dir, file)
                try:
                    gdf = gpd.read_file(shp_file)
                    gdf['政策类型'] = policy_type
                    if gdf.crs is None or gdf.crs.to_string() != 'EPSG:4326':
                        gdf = gdf.to_crs(epsg=4326)
                    policy_list.append(gdf)
                    print(f"已加载政策文件: {shp_file}")
                except Exception as e:
                    print(f"错误: 无法加载 {shp_file}")
                    print(f"错误信息: {e}")
except Exception as e:
    print(f"加载政策文件时发生错误: {e}")
    print("将继续不使用政策数据进行绘图")

# **合并政策数据**
gdf_policies_grouped = None
if policy_list:
    try:
        print("合并政策数据...")
        gdf_policies = gpd.GeoDataFrame(pd.concat(policy_list, ignore_index=True), crs="EPSG:4326")
        
        # 按政策类型分组并合并几何形状
        policy_groups = []
        for policy_type in policy_types.keys():
            subset = gdf_policies[gdf_policies['政策类型'] == policy_type]
            if not subset.empty:
                try:
                    # 尝试合并几何形状
                    union_geom = subset.unary_union
                    policy_groups.append({
                        'geometry': union_geom,
                        '政策类型': policy_type
                    })
                except Exception as e:
                    print(f"合并 {policy_type} 政策几何形状时出错: {e}")
        
        if policy_groups:
            gdf_policies_grouped = gpd.GeoDataFrame(policy_groups, crs="EPSG:4326")
            print("政策数据合并完成！")
        else:
            print("未能成功合并任何政策数据")
    except Exception as e:
        print(f"处理政策数据时发生错误: {e}")
        print("将继续不使用政策数据进行绘图")

print("开始对每种水坝类型和总变化进行数据处理和绘图...")

# **对每种水坝类型和总变化进行数据处理**
all_columns = []  # 存储所有规范化后的列名

# 处理每种水坝类型
for code, full_name in dam_type_mapping.items():
    change_col = f'change_{code}'
    norm_col = f'norm_{code}'
    all_columns.append(change_col)
    
    print(f"处理 {full_name} 水坝变化数据...")
    
    # 对非零变化值应用对数变换以便于可视化（保留符号）
    change_values = basins[change_col]
    # 使用 sign(x) * log(1+abs(x)) 来保留符号并处理可能的零值
    log_change = np.sign(change_values) * np.log1p(np.abs(change_values))
    
    # 将对数变换后的值规范化到 [-1, 1] 区间
    max_abs_val = np.abs(log_change).max()
    if max_abs_val > 0:
        basins[norm_col] = np.tanh(log_change / max_abs_val)
    else:
        basins[norm_col] = 0
        
    print(f"{full_name} 水坝变化值范围: [{basins[norm_col].min()}, {basins[norm_col].max()}]")

# 处理总变化
print("处理总变化数据...")
change_values = basins['change_total']
log_change = np.sign(change_values) * np.log1p(np.abs(change_values))
max_abs_val = np.abs(log_change).max()
if max_abs_val > 0:
    basins['norm_total'] = np.tanh(log_change / max_abs_val)
else:
    basins['norm_total'] = 0
all_columns.append('change_total')

print(f"总变化值范围: [{basins['change_total'].min()}, {basins['change_total'].max()}]")

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
        
        # 处理特殊情况：如果有专门的0区间
        if lower == 0 and upper == 0:
            # 计算等于0的数量
            count_zero = len(data[data[column] == 0])
            bin_counts.append(count_zero)
            bin_labels.append("= 0")
        elif lower < 0 and upper > 0:
            # 如果区间跨越0，则分成三部分: [lower, 0)、=0、(0, upper]
            
            # [lower, 0) 区间
            count_neg = len(data[(data[column] >= lower) & (data[column] < 0)])
            if count_neg > 0:
                bin_counts.append(count_neg)
                bin_labels.append(f"[{lower}, 0)")
            
            # = 0 值
            count_zero = len(data[data[column] == 0])
            if count_zero > 0:
                bin_counts.append(count_zero)
                bin_labels.append("= 0")
            
            # (0, upper] 区间
            count_pos = len(data[(data[column] > 0) & (data[column] <= upper)])
            if count_pos > 0:
                bin_counts.append(count_pos)
                bin_labels.append(f"(0, {upper}]")
        else:
            # 常规区间 [lower, upper]
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

# **绘制每种水坝类型的变化图**
print("开始绘制图表...")

# 为各类型水坝变化定义自定义边界，特别处理0值
custom_bins_mapping = {
    'change_e': np.array([-10000, -1000, -100, -50, -20, -5, -1, 0, 0, 1, 5, 20, 50, 100, 1000, 10000]),  # 土石坝
    'change_g': np.array([-5, -1, 0, 0, 1, 5, 10, 20, 50]),     # 重力坝
    'change_b': np.array([-50, -20, -10, -5, -1, 0, 0, 1, 5, 10, 20, 50]),     # 闸坝
    'change_a': np.array([-20, -10, -5, -2, -1, 0, 0, 1, 2, 5, 10, 20]),       # 拱坝
    'change_total': np.array([-10000, -1000, -100, -50, -20, -5, -1, 0, 0, 1, 5, 20, 50, 100, 1000, 10000])  # 总变化
}

for code, full_name in dam_type_mapping.items():
    print(f"绘制 {full_name} 水坝变化图...")
    change_col = f'change_{code}'

    # 创建绘图
    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.Robinson()})
    output_path = os.path.join(output_base_path, f'dam_change_{full_name}_2020-2015.png')
    
    # 获取该类型的自定义边界(如果有定义)
    custom_bins = custom_bins_mapping.get(change_col, None)
    
    # 根据数据分布调整自定义边界
    if custom_bins is not None:
        # 获取非零数据的最小值和最大值
        non_zero_data = basins[basins[change_col] != 0][change_col]
        if not non_zero_data.empty:
            data_min = non_zero_data.min()
            data_max = non_zero_data.max()
            
            # 确保自定义边界涵盖数据范围
            if data_min < custom_bins[0]:
                custom_bins[0] = data_min * 1.1  # 稍微扩展范围
            if data_max > custom_bins[-1]:
                custom_bins[-1] = data_max * 1.1  # 稍微扩展范围
    
    # 生成并保存区间统计
    csv_output_path = os.path.join(csv_output_dir, f'bin_statistics_{full_name}_2020-2015.csv')
    if custom_bins is not None:
        generate_bin_statistics(basins, change_col, custom_bins, csv_output_path)
    
    # 绘制并保存图像
    plot_map(
        data=basins,
        column=change_col,
        ax=ax,
        title=f'{full_name.title()} Dam Change (2020-2015)',
        output_path=output_path,
        gdf_policies_grouped=gdf_policies_grouped,
        custom_bins=custom_bins
    )

# **绘制总变化图**
print("绘制总变化图...")
fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.Robinson()})
output_path_total = os.path.join(output_base_path, 'dam_change_total_2020-2015.png')

# 获取总变化的自定义边界
custom_bins_total = custom_bins_mapping.get('change_total', None)

# 根据数据分布调整总变化的自定义边界
if custom_bins_total is not None:
    non_zero_data = basins[basins['change_total'] != 0]['change_total']
    if not non_zero_data.empty:
        data_min = non_zero_data.min()
        data_max = non_zero_data.max()
        
        # 确保自定义边界涵盖数据范围
        if data_min < custom_bins_total[0]:
            custom_bins_total[0] = data_min * 1.1
        if data_max > custom_bins_total[-1]:
            custom_bins_total[-1] = data_max * 1.1

# 生成并保存总变化的区间统计
csv_output_path_total = os.path.join(csv_output_dir, 'bin_statistics_total_2020-2015.csv')
if custom_bins_total is not None:
    generate_bin_statistics(basins, 'change_total', custom_bins_total, csv_output_path_total)

# 绘制并保存总变化图
plot_map(
    data=basins,
    column='change_total',
    ax=ax,
    title='Total Dam Change (2020-2015)',
    output_path=output_path_total,
    gdf_policies_grouped=gdf_policies_grouped,
    custom_bins=custom_bins_total
)

print("所有图像绘制完成！")