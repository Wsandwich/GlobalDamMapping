import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

# 文件路径
basins_shp_path = "/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/BasinATLAS_Data_v10_shp/BasinATLAS_v10_shp/BasinATLAS_v10_lev03.shp"
dof_json_path = "/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/wyj/大坝知识图谱材料/大坝知识图谱材料/代码/Free-Flowing-Rivers/output/dof_comparison_20250426_181552.json"
output_dir = "/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/wyj/大坝知识图谱材料/大坝知识图谱材料/代码/Free-Flowing-Rivers/output/maps"
os.makedirs(output_dir, exist_ok=True)

# 加载流域数据
print("Loading basin data...")
basins = gpd.read_file(basins_shp_path)
print(f"Loaded {len(basins)} basins")

# 加载DOF比较结果
print("Loading DOF comparison results...")
with open(dof_json_path, 'r') as f:
    dof_results = json.load(f)
print(f"Loaded {len(dof_results)} DOF comparison results")

# 创建映射字典
dof_dict = {}
for file_name, values in dof_results.items():
    hybas_id = int(Path(file_name).stem)
    if values["difference"] is not None:
        dof_dict[hybas_id] = values["difference"]

# 映射DOF变化
print("Mapping DOF changes to basins...")
basins['dof_change'] = basins['HYBAS_ID'].map(dof_dict)
basins_with_dof = basins.dropna(subset=['dof_change'])
print(f"Number of basins with DOF data: {len(basins_with_dof)}")
min_dof = basins_with_dof['dof_change'].min()
max_dof = basins_with_dof['dof_change'].max()
print(f"DOF change range: {min_dof:.4f} to {max_dof:.4f}")

# 检查数据范围是否有NaN或Inf
if np.isnan(min_dof) or np.isnan(max_dof) or np.isinf(min_dof) or np.isinf(max_dof):
    raise ValueError("min_dof or max_dof contains NaN or Inf")

# 过滤掉-0.1到0.1之间的流域（视为无变化）
basins_with_significant_dof = basins_with_dof[
    (basins_with_dof['dof_change'] < -0.1) | (basins_with_dof['dof_change'] > 0.1)
]
print(f"Number of basins with significant DOF change: {len(basins_with_significant_dof)}")

#fee2bb, #eeb882, #df8c50, #cf5b27, #be0f0a);

#FEE2BB
#FC8C59
#BE0F0A

#fee2bb, #fdd4a5, #fcc590, #fcb57d, #fca46c, #f7945d, #f28450, #ed7344, #e25f35, #d64b26, #ca3318, #be0f0a);

# 定义分段颜色映射（移除白色）
colors = [
    '#007560',  # medium blue (decreased fragmentation)
    '#659B91',  # light blue
    '#B5D4C1',  # very light blue
    '#f7f7f7',  # 移除白色（无变化区域将显示底图）
    '#fee2bb',  # very light red (increased fragmentation)
    '#fcb57d',  # light red
    '#FC8C59',  # medium red
    '#e25f35',  # dark red
    '#be0f0a'   # very dark red
]


custom_cmap = ListedColormap(colors)

# 定义颜色区间边界（移除-0.1到0.1的区间）
lower_bound = min(min_dof, -5)
upper_bound = max(max_dof, 50)
boundaries = [lower_bound, -5, -1, -0.1, 0.1, 1, 5, 10, 15, upper_bound]
norm = BoundaryNorm(boundaries, custom_cmap.N, clip=True)

# 绘制DOF变化地图
print("Creating DOF change map with 50m base map...")
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
ax.set_global()  # Ensure the map shows the entire globe

# 添加50m分辨率的Natural Earth底图

hatch_pattern='////////////'
hatch_color='#DED9E3'
hatch_linewidth=0.4


# Set global hatching properties
plt.rcParams['hatch.color'] = hatch_color
plt.rcParams['hatch.linewidth'] = hatch_linewidth

land = cfeature.NaturalEarthFeature(
    'physical', 'land', '10m',
    facecolor='#ffffff',
    # edgecolor=hatch_color,  # Set the hatch color here
    hatch=hatch_pattern,  # 从左上到右下的斜线
    # linewidth=hatch_linewidth  # This will affect the hatch linewidth
    zorder=5
)
ax.add_feature(land)

# 绘制河流（使用Natural Earth 50m河流数据）
rivers = cfeature.NaturalEarthFeature(
    category='physical',
    name='rivers_lake_centerlines',
    scale='50m',
    facecolor='none',
    edgecolor=[151/255, 182/255, 225/255]  # RGB标准化
)
ax.add_feature(rivers, linewidth=0.5, zorder=11)

# 绘制湖泊
ax.add_feature(cfeature.LAKES, alpha=1.0, facecolor=[151/255, 182/255, 225/255], zorder=11)

# 绘制流域边界（所有流域）
# basins_with_significant_dof.boundary.plot(ax=ax, linewidth=0.3, color='black', linestyle='--', transform=ccrs.PlateCarree(), zorder=12)
basins_with_significant_dof.boundary.plot(ax=ax, linewidth=0.3, color='black', transform=ccrs.PlateCarree(), zorder=12)

# 绘制显著DOF变化的流域（|DOF| > 0.1）
if len(basins_with_significant_dof) > 0:
    basins_with_significant_dof.plot(
        column='dof_change', 
        ax=ax, 
        cmap=custom_cmap, 
        norm=norm, 
        legend=False, 
        transform=ccrs.PlateCarree(), 
        zorder=10
    )

ax.spines[:].set_visible(False)

# 添加色条
sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.8, pad=0.05, extend='both')
cbar.set_label('DOF Change (2020 - 2010)', fontsize=12)

# 设置色条刻度
ticks = [lower_bound, -5, -1, -0.1, 0.1, 1, 10, 30, 50, upper_bound]
cbar.set_ticks(ticks)
cbar.set_ticklabels([f"{t:.1f}" if t != lower_bound and t != upper_bound else f"<{t:.1f}" if t == lower_bound else f">{t:.1f}" for t in ticks])

# 添加图例说明
legend_text = ("Negative values (Blue) = Decreased fragmentation (Improvement)\n"
               "Positive values (Red) = Increased fragmentation (Deterioration)\n"
               "No color (-0.1 to 0.1) = No significant change (Base map visible)")
plt.figtext(0.5, 0.01, legend_text, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# 添加标题和统计信息
plt.title('Basin Fragmentation (DOF) Change 2010-2020 with 50m Base Map', fontsize=16)
plt.text(0.01, 0.01, f'Basins with data: {len(basins_with_dof)} / {len(basins)}', 
         transform=ax.transAxes, fontsize=10)

improved = sum(1 for v in basins_with_dof['dof_change'] if v < -0.1)
worsened = sum(1 for v in basins_with_dof['dof_change'] if v > 0.1)
unchanged = sum(1 for v in basins_with_dof['dof_change'] if -0.1 <= v <= 0.1)
stats_text = f"Improved basins: {improved} ({improved/len(basins_with_dof)*100:.1f}%)\n"
stats_text += f"Deteriorated basins: {worsened} ({worsened/len(basins_with_dof)*100:.1f}%)\n"
stats_text += f"Unchanged basins: {unchanged} ({unchanged/len(basins_with_dof)*100:.1f}%)"
# plt.text(0.01, 0.05, stats_text, transform=ax.transAxes, fontsize=10,
#          bbox=dict(facecolor='white', alpha=0.8))

# 保存地图
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
map_output_path = os.path.join(output_dir, f'dof_change_map_with_50m_basemap_{current_time}.png')
plt.savefig(map_output_path, dpi=500, bbox_inches='tight')
print(f"DOF change map saved to {map_output_path}")
plt.close()

print("Map with 50m base map and segmented color mapping completed!")