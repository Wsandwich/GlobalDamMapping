import geopandas as gpd
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import cartopy.crs as ccrs

# **绘图函数**
def plot_map(data, column, ax, title, output_path, gdf_policies_grouped):
    # **分离无变化和有变化的流域**
    data_no_change = data[data[column] == 0]
    data_change = data[data[column] != 0]
    
    # **添加全球海岸线**
    ax.coastlines(resolution='50m', linewidth=0.3, color="black")
    
    # **绘制无变化的流域（灰色填充）**
    data_no_change.plot(ax=ax, color='whitesmoke', edgecolor='none',
                        transform=ccrs.PlateCarree(), label='No Change')
    
    # **绘制有变化的流域**
    data_change.plot(column=column, ax=ax, cmap='RdBu_r',
                     edgecolor='none', vmin=-1, vmax=1,
                     legend=False, label='Change Density',
                     transform=ccrs.PlateCarree())
    
    # **绘制合并后的政策线**
    if gdf_policies_grouped is not None:
        for policy_type, color in policy_types.items():
            policy_data = gdf_policies_grouped[gdf_policies_grouped['政策类型'] == policy_type]
            if not policy_data.empty:
                policy_data.plot(
                    ax=ax, edgecolor=color, linewidth=1.5, facecolor='none',
                    transform=ccrs.PlateCarree()
                )
    
    ax.set_global()
    ax.gridlines(draw_labels=False, visible=False)
    # 去除图框
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # **设置标题**
    ax.set_title(title, fontsize=12)
    
    # **添加图例**
    plt.legend()
    
    # **保存图像**
    plt.savefig(output_path, dpi=2000, bbox_inches='tight')
    plt.close()

# **文件路径**
shp_path = r'E:\wyj\project\dam\result1\output\BasinATLAS_v10_lev06_with_dam_stats.shp'
policy_folder = r'E:\wyj\project\dam\draw\数据\政策'
output_base_path = r'E:\wyj\dam\v4\image\result2\images_log'

# **确保输出目录存在**
os.makedirs(output_base_path, exist_ok=True)

# **读取流域 Shapefile**
print("读取流域 Shapefile...")
basins = gpd.read_file(shp_path)

# **计算四种水坝类型的 dam_change（2020 - 2010）**
dam_types = ['e', 'g', 'b', 'a']  # 使用第一个代码中的简化列名
for dam_type in dam_types:
    basins[f'{dam_type}_change'] = basins[f'2020_{dam_type}'] - basins[f'2010_{dam_type}']

# **计算总变化**
basins['dam_change_total'] = basins[[f'{dam_type}_change' for dam_type in dam_types]].sum(axis=1)

# **转换坐标系到 WGS84（EPSG:4326）**
basins.set_crs(epsg=4326, inplace=True)

# **遍历 "拆除" 和 "建设" 文件夹，读取政策文件**
policy_types = {"拆除_remove_Guyane": "purple", "建设": "green"}
policy_list = []

print("加载政策文件...")
for policy_type in policy_types.keys():
    policy_dir = os.path.join(policy_folder, policy_type)
    if not os.path.exists(policy_dir):
        print(f"警告: 目录 {policy_dir} 不存在，跳过。")
        continue
    for file in os.listdir(policy_dir):
        if file.endswith(".shp"):
            shp_file = os.path.join(policy_dir, file)
            try:
                gdf = gpd.read_file(shp_file)
                gdf['政策类型'] = policy_type
                gdf = gdf.to_crs(epsg=4326)
                policy_list.append(gdf)
            except Exception as e:
                print(f"错误: 无法加载 {shp_file}")
                print(f"错误信息: {e}")


# **合并政策数据**
if policy_list:
    gdf_policies = gpd.GeoDataFrame(pd.concat(policy_list, ignore_index=True), crs="EPSG:4326")
    print("政策 Shapefile 加载完成！")
else:
    gdf_policies = None
    print("未找到任何政策 Shapefile。")

# **合并同一类型的政策线**
if gdf_policies is not None:
    print("合并政策线...")
    policy_groups = []
    print("政策数据统计：")
    print(gdf_policies['政策类型'].value_counts())
    print("各政策类型的几何形状数量：")
    for policy_type in policy_types.keys():
        subset = gdf_policies[gdf_policies['政策类型'] == policy_type]
        print(f"{policy_type}: {len(subset)} 个几何形状")
        if not subset.empty:
            print(f"  几何形状类型: {subset.geometry.geom_type.value_counts()}")
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
        print("政策线合并完成！")
    else:
        print("未能成功合并任何政策数据")
        gdf_policies_grouped = None
else:
    gdf_policies_grouped = None

print("开始绘图...")
print(dam_types)

# **为每种水坝类型和总变化绘图**
for dam_type in dam_types:
    print(f"绘制 {dam_type.capitalize()} 图...")
    
    # 将变化值转换为正值并避免 log(0) 或负值问题
    change_values = basins[f'{dam_type}_change']
    # 使用 np.sign 保留符号，np.log1p 处理绝对值（log1p(x) = log(1+x) 避免 log(0)）
    log_change = np.sign(change_values) * np.log1p(np.abs(change_values))

    # **归一化 dam_change 到 [-1, 1]**

    max_abs = np.abs(log_change).max()
    if max_abs > 0:
        basins[f'{dam_type}_density_sc'] = np.tanh(log_change / max_abs)
    else:
        basins[f'{dam_type}_density_sc'] = 0
    
    # **创建绘图**
    fig, ax = plt.subplots(figsize=(20, 16), subplot_kw={'projection': ccrs.Robinson()})
    output_path = os.path.join(output_base_path, f'dam_change_{dam_type}_2020-2010.png')
    plot_map(basins, f'{dam_type}_density_sc', ax, f'{dam_type.capitalize()} Dam Change (2020-2010)', output_path, gdf_policies_grouped)
    print(f"图像已保存至: {output_path}")

# **总变化图**
max_abs_total = basins['dam_change_total'].abs().max()
if max_abs_total > 0:
    basins['density_sc_total'] = np.tanh(basins['dam_change_total'] / max_abs_total)
else:
    basins['density_sc_total'] = 0

fig, ax = plt.subplots(figsize=(20, 16), subplot_kw={'projection': ccrs.Robinson()})
output_path_total = os.path.join(output_base_path, 'dam_change_total_2020-2010.png')
plot_map(basins, 'density_sc_total', ax, 'Total Dam Change (2020-2010)', output_path_total)
print(f"图像已保存至: {output_path_total}")