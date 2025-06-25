import geopandas as gpd
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from shapely.geometry import LineString, MultiLineString
import cartopy.crs as ccrs
import fiona
import cartopy.feature as cfeature
from datetime import datetime
import time

# Define base directory
base_dir = r"/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/dof_result"
base_dir_2010 = r"/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/dof_result_2010"

output_path = os.path.join(base_dir, "output", "draw")
os.makedirs(output_path, exist_ok=True)

# 2020年和2010年的文件列表
gpkg_files_2020 = ['/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/dof_result/7030047060.gpkg']

gpkg_files_2010 = []

for file in gpkg_files_2020:
    file_2010 = os.path.join(base_dir_2010, os.path.basename(file))
    gpkg_files_2010.append(file_2010)


# 蓝色表示负向变化（河流状况变差），红色表示正向变化（河流状况变好）
colors_diff = ["darkblue", "blue", "lightblue", "white", "yellow", "orange", "red"]
colors_diff = ["darkblue", "blue", "lightblue", "white", "yellow", "orange", "red"]
cmap_diff = LinearSegmentedColormap.from_list("diff_colormap", colors_diff, N=200)
norm_diff = Normalize(vmin=-100, vmax=100)  # 范围从-100到100

# Width mapping function based on discharge
def width_logic(riv_ord):
    
    # print(f"Processing river order {riv_ord}")
    # print(f'The type of riv_ord is {type(riv_ord)}')
    river_width_dict = {}
    river_width_dict[1] = 0.9
    river_width_dict[2] = 0.8
    river_width_dict[3] = 0.7
    river_width_dict[4] = 0.6
    river_width_dict[5] = 0.3
    river_width_dict[6] = 0.2
    river_width_dict[7] = 0.05
    # river_width_dict[8] = 0.1
    # river_width_dict[9] = 0.05
    # river_width_dict[10] = 0.05

    return river_width_dict.get(riv_ord, 0.05)

# 函数：简化并投影几何形状
def simplify_and_project_geometry(geom, tolerance=0.01, source_crs="EPSG:4326", target_proj=ccrs.Robinson()):
    """简化并投影几何形状"""
    if geom is None:
        return None
    
    # 首先简化几何形状
    if isinstance(geom, LineString):
        geom = geom.simplify(tolerance)
    elif isinstance(geom, MultiLineString):
        simplified_parts = [line.simplify(tolerance) for line in geom.geoms]
        geom = MultiLineString(simplified_parts)
    
    # 然后转换投影
    transformer = target_proj.transform_points
    
    if isinstance(geom, LineString):
        pts = np.array(geom.coords)
        if pts.size > 0:
            transformed_pts = transformer(ccrs.PlateCarree(), pts[:, 0], pts[:, 1])
            return LineString(transformed_pts[:, 0:2])
    elif isinstance(geom, MultiLineString):
        transformed_lines = []
        for line in geom.geoms:
            pts = np.array(line.coords)
            if pts.size > 0:
                transformed_pts = transformer(ccrs.PlateCarree(), pts[:, 0], pts[:, 1])
                transformed_lines.append(LineString(transformed_pts[:, 0:2]))
        return MultiLineString(transformed_lines)
    
    return geom

# 函数：读取2020年GeoPackage文件数据
def read_gpkg_2020(gpkg_file):
    """从2020年GPKG文件中读取数据"""
    file_name = os.path.basename(gpkg_file)
    print(f"Processing 2020 file: {file_name}")
    
    try:
        # 检查文件是否存在
        if not os.path.exists(gpkg_file):
            print(f"  - File {file_name} does not exist.")
            return None
            
        # 读取DOF图层，包含所有需要的字段
        start_time = time.time()
        dof_gdf = gpd.read_file(gpkg_file, layer='DOF', usecols=['geometry', 'GOID', 'DOF', 'DIS_AV_CMS', 'dof_all', 'RIV_ORD'])
        elapsed_time = time.time() - start_time
        print(f"  - DOF layer records: {len(dof_gdf)} (took {elapsed_time:.2f} seconds)")
        
        # 处理GOID确保作为字符串类型
        if 'GOID' in dof_gdf.columns:
            dof_gdf['GOID'] = dof_gdf['GOID'].astype(str)
        else:
            print(f"  - Warning: No GOID column found in {file_name}")
            return None
        
        # 处理dof_all字段
        if 'dof_all' in dof_gdf.columns:
            dof_gdf['dof_all'] = pd.to_numeric(dof_gdf['dof_all'], errors='coerce').fillna(0)
            
            # 添加基于流量的线宽
            if 'RIV_ORD' in dof_gdf.columns:
                start_time = time.time()
                # 过滤掉小流量河流
                dof_gdf = dof_gdf[dof_gdf['RIV_ORD'] <= 7]
                dof_gdf['width'] = dof_gdf['RIV_ORD'].apply(width_logic)
                elapsed_time = time.time() - start_time
                print(f"  - Applied width_logic and filtered low discharge (took {elapsed_time:.2f} seconds)")
            else:
                dof_gdf['width'] = 0.5  # 默认宽度
            
            # 过滤非MultiLineString几何
            valid_dof_gdf = dof_gdf[dof_gdf.geometry.apply(lambda x: isinstance(x, MultiLineString))]
            
            # 确保GeoDataFrame有CRS
            if valid_dof_gdf.crs is None:
                valid_dof_gdf.crs = "EPSG:4326"  # 如果缺失，设置为WGS 84
            
            
            return valid_dof_gdf
            
        else:
            print(f"  - No dof_all field found in DOF layer of {file_name}")
            return None
            
    except Exception as e:
        print(f"Error processing 2020 file {file_name}: {str(e)}")
        return None

# 函数：读取2010年GeoPackage文件数据（只读取GOID和dof_all）
def read_gpkg_2010(gpkg_file):
    """从2010年GPKG文件中只读取GOID和dof_all"""
    file_name = os.path.basename(gpkg_file)
    print(f"Processing 2010 file: {file_name}")
    
    try:
        # 检查文件是否存在
        if not os.path.exists(gpkg_file):
            print(f"  - File {file_name} does not exist.")
            return None
            
        # 读取DOF图层，只包含GOID和dof_all字段
        start_time = time.time()
        dof_gdf = gpd.read_file(gpkg_file, layer='DOF', usecols=['GOID', 'dof_all'])
        elapsed_time = time.time() - start_time
        print(f"  - 2010 DOF layer records: {len(dof_gdf)} (took {elapsed_time:.2f} seconds)")
        
        # 处理GOID确保作为字符串类型
        if 'GOID' in dof_gdf.columns:
            dof_gdf['GOID'] = dof_gdf['GOID'].astype(str)
        else:
            print(f"  - Warning: No GOID column found in 2010 file {file_name}")
            return None
        
        # 处理dof_all字段
        if 'dof_all' in dof_gdf.columns:
            dof_gdf['dof_all'] = pd.to_numeric(dof_gdf['dof_all'], errors='coerce').fillna(0)
            return dof_gdf
        else:
            print(f"  - No dof_all field found in 2010 DOF layer of {file_name}")
            return None
            
    except Exception as e:
        print(f"Error processing 2010 file {file_name}: {str(e)}")
        return None

# 函数：基于GOID计算两个时期之间的dof_all差值
def calculate_dof_diff_by_goid(gdf_2020, gdf_2010_data):
    """使用GOID匹配计算2020年和2010年之间的dof_all差值"""
    if gdf_2020 is None:
        print("No 2020 data available for comparison")
        return None
        
    if gdf_2010_data is None:
        print("No 2010 data found, assuming all 2010 dof_all values are 0")
        # 复制2020年数据，将dof_all直接作为差值
        diff_gdf = gdf_2020.copy()
        diff_gdf['dof_diff'] = diff_gdf['dof_all']
        return diff_gdf
    
    # 创建差值GeoDataFrame
    diff_gdf = gdf_2020.copy()
    diff_gdf['dof_diff'] = 0  # 默认差值为0
    
    # 从2010年数据创建GOID到dof_all的映射
    dof_2010_map = dict(zip(gdf_2010_data['GOID'], gdf_2010_data['dof_all']))
    
    # 计算差值
    start_time = time.time()
    for idx, row in diff_gdf.iterrows():
        goid = row['GOID']
        if goid in dof_2010_map:
            # 如果在2010年数据中找到匹配的GOID，计算差值
            diff_gdf.loc[idx, 'dof_diff'] = row['dof_all'] - dof_2010_map[goid]
        else:
            # 如果在2010年数据中未找到匹配的GOID，差值就是2020年的值
            diff_gdf.loc[idx, 'dof_diff'] = row['dof_all']
    
    elapsed_time = time.time() - start_time
    print(f"  - Calculated dof_diff for {len(diff_gdf)} records (took {elapsed_time:.2f} seconds)")
    print(f"  - Matched {len([g for g in diff_gdf['GOID'] if g in dof_2010_map])} records with 2010 data")
    print(f"  - Could not match {len([g for g in diff_gdf['GOID'] if g not in dof_2010_map])} records")
    
    return diff_gdf

# 函数：绘制河流差异图
def plot_dof_diff_map(output_file, diff_gdf, title="River Fragmentation Change (2010-2020)", hatch_pattern='////////////', hatch_color='#DDD4DA', hatch_linewidth=0.2):
    """绘制河流dof_all差异地图"""
    if diff_gdf is None or len(diff_gdf) == 0:
        print("No difference data to plot")
        return
    
    print("开始绘制河流差异图...")
    
    # Set global hatching properties
    plt.rcParams['hatch.color'] = hatch_color
    plt.rcParams['hatch.linewidth'] = hatch_linewidth

    # 创建带有 Robinson 投影的画布
    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.Robinson()})
    # 添加灰色陆地底图
    has_world_map = True
    if has_world_map:
        land = cfeature.NaturalEarthFeature(
            'physical', 'land', '10m',
            facecolor='#f0f0f0',
            # edgecolor=hatch_color,  # Set the hatch color here
            hatch=hatch_pattern,  # 从左上到右下的斜线
            # linewidth=hatch_linewidth  # This will affect the hatch linewidth
        )
        ax.add_feature(land)

    else:
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)

    
    # 创建线段集合用于绘制河流差异
    segments = []
    color_values = []
    widths = []
    
    start_time = time.time()
    for idx, row in diff_gdf.iterrows():
        try:
            # 使用投影后的几何形状
            geom = row['projected_geom'] if 'projected_geom' in row else row.geometry
            
            if isinstance(geom, LineString):
                coords = list(geom.coords)
                segments.append(coords)
                color_values.append(row['dof_diff'])  # 使用差值作为颜色值
                widths.append(row['width'])
            elif isinstance(geom, MultiLineString):
                for line in geom.geoms:
                    coords = list(line.coords)
                    segments.append(coords)
                    color_values.append(row['dof_diff'])  # 使用差值作为颜色值
                    widths.append(row['width'])
        except Exception as e:
            print(f"Error processing geometry at index {idx}: {e}")
    
    print(f"处理线段完成，共 {len(segments)} 条线段，耗时: {time.time() - start_time:.2f}秒")
    
    # 绘制河流线段
    start_time = time.time()
    if segments:
        lc = LineCollection(
            segments,
            linewidths=widths,
            cmap=cmap_diff,  # 使用差值颜色映射
            norm=norm_diff,  # 使用差值规范化
            # transform=ccrs.PlateCarree() if 'projected_geom' not in diff_gdf.columns else None,
            transform=ccrs.PlateCarree(),
            zorder=10  # 显式设置高 Z-order
        )
        lc.set_array(np.array(color_values))
        ax.add_collection(lc)
    
    print(f"绘制线段完成，耗时: {time.time() - start_time:.2f}秒")
    

    print(f"dof_diff range: min={diff_gdf['dof_diff'].min()}, max={diff_gdf['dof_diff'].max()}")
    print(f"dof_diff value counts: {diff_gdf['dof_diff'].value_counts()}")


    # 设置地图范围为全球
    ax.set_global()
    
    # 添加网格线
    ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    
    # 设置标题
    ax.set_title(title, fontsize=16, pad=20)
    plt.suptitle("River Fragmentation Change Analysis", fontsize=18, y=0.95)
    
    # 添加色条
    sm = ScalarMappable(cmap=cmap_diff, norm=norm_diff)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.04, pad=0.1)
    cbar.set_label('DOF Change (-100 to +100)', fontsize=12)
    
    # 添加数据源说明
    plt.figtext(0.5, 0.02, "Data source: River fragmentation analysis 2010-2020", ha='center', fontsize=10)
    
    # 保存图像
    start_time = time.time()
    plt.savefig(output_file, dpi=2000, bbox_inches='tight')
    print(f"差异图已保存到 {output_file}，耗时: {time.time() - start_time:.2f}秒")
    plt.close(fig)

# 主函数
def main():
    # 获取当前时间并格式化为字符串
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("开始处理2020年和2010年数据比较...")
    all_diff_data = []
    
    # 为每对文件计算差值并绘图
    for i, gpkg_file_2020 in enumerate(gpkg_files_2020):
        # 获取文件名
        file_name = os.path.basename(gpkg_file_2020)
        
        # 读取2020年数据（包含完整的空间和属性信息）
        gdf_2020 = read_gpkg_2020(gpkg_file_2020)
        if gdf_2020 is None:
            print(f"无法读取2020年文件: {file_name}，跳过此文件的比较")
            continue
        
        # 检查是否有对应的2010年文件
        gpkg_file_2010 = None
        if i < len(gpkg_files_2010):
            gpkg_file_2010 = gpkg_files_2010[i]
        
        # 读取2010年数据（如果存在，只读取GOID和dof_all）
        gdf_2010_data = None
        if gpkg_file_2010 and os.path.exists(gpkg_file_2010):
            print(f"读取对应的2010年数据: {os.path.basename(gpkg_file_2010)}")
            gdf_2010_data = read_gpkg_2010(gpkg_file_2010)
        else:
            print(f"未找到对应的2010年数据文件，假设所有2010年dof_all值为0")
        
        # 基于GOID计算差值
        diff_gdf = calculate_dof_diff_by_goid(gdf_2020, gdf_2010_data)
        
        if diff_gdf is not None and not diff_gdf.empty:
            all_diff_data.append(diff_gdf)
            
            # 为单个文件绘制差异图
            output_file = os.path.join(output_path, f"dof_diff_{file_name.replace('.gpkg', '')}_{current_time}.png")
            plot_dof_diff_map(output_file, diff_gdf, f"River Fragmentation Change (2010-2020) - {file_name}")
    
    # 如果有多个文件，绘制合并的差异图
    if len(all_diff_data) > 1:
        print(f"合并 {len(all_diff_data)} 个差异数据集绘制总体差异图...")
        combined_output_file = os.path.join(output_path, f"global_dof_diff_{current_time}.png")
        # 使用GeoDataFrame的concat方法合并所有差异数据
        combined_diff_gdf = pd.concat(all_diff_data)
        plot_dof_diff_map(combined_output_file, combined_diff_gdf, "Global River Fragmentation Change (2010-2020)")
    
    print("差异分析完成!")

if __name__ == "__main__":
    main()