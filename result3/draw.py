import geopandas as gpd
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from shapely.geometry import LineString

# 定义基础目录
base_dir = r"/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/dof_result"  # 使用你提供的路径
output_path = os.path.join(base_dir, "output", "draw")
os.makedirs(output_path, exist_ok=True)

# 查找所有gpkg文件
gpkg_files = glob.glob(os.path.join(base_dir, "*.gpkg"))
print(f"找到 {len(gpkg_files)} 个GPKG文件")

# 设置颜色映射
colors = ["darkblue", "aqua", "yellow", "orange", "red"]
cmap = LinearSegmentedColormap.from_list("custom_colormap", colors, N=100)
norm = Normalize(vmin=0, vmax=100)

# 线宽映射函数
def width_logic(dis_av_cms):
    if pd.isna(dis_av_cms):  # 处理NaN值
        return 0.5
    
    if dis_av_cms < 600:
        return 0.5
    elif 600 <= dis_av_cms < 1500:
        return 1
    elif 1500 <= dis_av_cms < 2000:
        return 2
    else:
        return 4

# 创建两个图 - 一个用于DOF，一个用于dof_all
fig1, ax1 = plt.subplots(figsize=(15, 12))
fig2, ax2 = plt.subplots(figsize=(15, 12))

# 读取所有文件并合并数据
all_dof_gdf_list = []
all_dof_all_gdf_list = []

import pandas as pd

for gpkg_file in gpkg_files:
    file_name = os.path.basename(gpkg_file)
    print(f"处理文件: {file_name}")
    
    try:
        # 读取DOF图层
        try:
            dof_gdf = gpd.read_file(gpkg_file, layer='DOF')
            print(f"  - DOF图层记录数: {len(dof_gdf)}")
            
            # 确保DOF字段存在并转为数值
            if 'DOF' in dof_gdf.columns:
                dof_gdf['DOF'] = pd.to_numeric(dof_gdf['DOF'], errors='coerce').fillna(0)
                
                # 添加线宽
                if 'DIS_AV_CMS' in dof_gdf.columns:
                    dof_gdf['width'] = dof_gdf['DIS_AV_CMS'].apply(width_logic)
                else:
                    dof_gdf['width'] = 0.5  # 默认线宽
                
                # 过滤掉非LineString几何对象
                valid_dof_gdf = dof_gdf[dof_gdf.geometry.apply(lambda x: isinstance(x, LineString))]
                if len(valid_dof_gdf) < len(dof_gdf):
                    print(f"  - 警告: 移除了 {len(dof_gdf) - len(valid_dof_gdf)} 个非LineString几何对象")
                
                all_dof_gdf_list.append(valid_dof_gdf)
            else:
                print(f"  - 警告: DOF字段在{file_name}中不存在")
        except Exception as e:
            print(f"  - 读取DOF图层出错: {str(e)}")
        
        # 检查和读取dof_all字段
        # 如果dof_all字段在DOF图层中
        if 'dof_all' in dof_gdf.columns:
            dof_gdf_copy = dof_gdf.copy()
            dof_gdf_copy['dof_all'] = pd.to_numeric(dof_gdf_copy['dof_all'], errors='coerce').fillna(0)
            
            # 过滤掉非LineString几何对象
            valid_dof_all_gdf = dof_gdf_copy[dof_gdf_copy.geometry.apply(lambda x: isinstance(x, LineString))]
            all_dof_all_gdf_list.append(valid_dof_all_gdf)
        else:
            # 尝试读取其他图层中的dof_all字段
            try:
                # 列出所有图层
                layers = fiona.listlayers(gpkg_file)
                found_dof_all = False
                
                for layer in layers:
                    if layer == 'DOF':  # 已经处理过
                        continue
                    
                    try:
                        layer_gdf = gpd.read_file(gpkg_file, layer=layer)
                        if 'dof_all' in layer_gdf.columns:
                            print(f"  - 在图层 '{layer}' 中找到 dof_all 字段")
                            layer_gdf['dof_all'] = pd.to_numeric(layer_gdf['dof_all'], errors='coerce').fillna(0)
                            
                            # 添加线宽
                            if 'DIS_AV_CMS' in layer_gdf.columns:
                                layer_gdf['width'] = layer_gdf['DIS_AV_CMS'].apply(width_logic)
                            else:
                                layer_gdf['width'] = 0.5  # 默认线宽
                            
                            # 过滤掉非LineString几何对象
                            valid_layer_gdf = layer_gdf[layer_gdf.geometry.apply(lambda x: isinstance(x, LineString))]
                            if len(valid_layer_gdf) < len(layer_gdf):
                                print(f"  - 警告: 在图层 '{layer}' 中移除了 {len(layer_gdf) - len(valid_layer_gdf)} 个非LineString几何对象")
                            
                            all_dof_all_gdf_list.append(valid_layer_gdf)
                            found_dof_all = True
                            break
                    except Exception as e:
                        print(f"  - 读取图层 '{layer}' 出错: {str(e)}")
                
                if not found_dof_all:
                    print(f"  - 警告: 在{file_name}的所有图层中未找到dof_all字段")
            except Exception as e:
                print(f"  - 获取图层列表出错: {str(e)}")
                
    except Exception as e:
        print(f"处理文件 {file_name} 时出错: {str(e)}")

# 如果需要导入fiona库
import fiona

# 检查是否有数据用于绘图
if not all_dof_gdf_list:
    print("未找到可用的DOF数据，无法绘制DOF图")
else:
    print(f"合并 {len(all_dof_gdf_list)} 个数据集用于DOF图")

if not all_dof_all_gdf_list:
    print("未找到可用的dof_all数据，无法绘制dof_all图")
else:
    print(f"合并 {len(all_dof_all_gdf_list)} 个数据集用于dof_all图")

# 定义绘图函数
def plot_rivers(ax, gdf_list, title, dof_col):
    # 创建线段集合
    segments = []
    colors_array = []
    widths = []
    
    for gdf in gdf_list:
        for idx, row in gdf.iterrows():
            try:
                # 确保几何对象是LineString
                if isinstance(row.geometry, LineString):
                    coords = list(row.geometry.coords)
                    segments.append(coords)
                    colors_array.append(row[dof_col])
                    widths.append(row['width'])
            except (AttributeError, TypeError) as e:
                print(f"处理几何对象时出错 (index {idx}): {e}")
    
    # 绘制线段集合
    if segments:
        lc = LineCollection(segments, linewidths=widths, cmap=cmap, norm=norm)
        lc.set_array(colors_array)  # 绑定颜色数据
        ax.add_collection(lc)
    
    # 设置轴属性
    ax.set_title(title, fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    
    # 自动调整视图以适应数据
    ax.autoscale_view()
    
    # 添加色标条
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.05, pad=0.05)
    cbar.set_label(f'{dof_col} Score (0-100)')
    
    return ax

# 绘制DOF图
if all_dof_gdf_list:
    plot_rivers(ax1, all_dof_gdf_list, "DOF Values for All Rivers", 'DOF')
    fig1.suptitle("River Fragmentation - DOF Values", fontsize=18)
    fig1.savefig(os.path.join(output_path, "all_DOF.png"), dpi=300, bbox_inches='tight')
    print(f"DOF图保存到 {os.path.join(output_path, 'all_DOF.png')}")

# 绘制dof_all图
if all_dof_all_gdf_list:
    plot_rivers(ax2, all_dof_all_gdf_list, "dof_all Values for All Rivers", 'dof_all')
    fig2.suptitle("River Fragmentation - dof_all Values", fontsize=18)
    fig2.savefig(os.path.join(output_path, "all_dof_all.png"), dpi=300, bbox_inches='tight')
    print(f"dof_all图保存到 {os.path.join(output_path, 'all_dof_all.png')}")

print("绘图完成！")