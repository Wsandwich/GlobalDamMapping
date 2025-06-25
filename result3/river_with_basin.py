import geopandas as gpd
import os
import pandas as pd
from pathlib import Path

# 文件路径
basins_shp_path = "/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/BasinATLAS_Data_v10_shp/BasinATLAS_v10_shp/BasinATLAS_v10_lev03.shp"
river_gpkg_path = "/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/FFR_river_network.gdb"

# 输出目录
output_dir = "/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/rivers_by_basin_03"

os.makedirs(output_dir, exist_ok=True)

# 加载流域和河流数据
print("正在加载流域数据...")
basins = gpd.read_file(basins_shp_path)
print("正在加载河流数据...")
rivers = gpd.read_file(river_gpkg_path)

# 检查并统一坐标参考系统（CRS）
if basins.crs != rivers.crs:
    print(f"流域 CRS: {basins.crs}, 河流 CRS: {rivers.crs}")
    print("正在将河流数据转换为流域的 CRS...")
    rivers = rivers.to_crs(basins.crs)

# 确定流域的唯一标识字段（请根据实际数据调整）
# 假设流域有一个字段如 'BASIN_ID' 或 'HYBAS_ID'（BasinATLAS 常用）
id_field = 'HYBAS_ID'  # 请替换为实际字段名，例如 'BASIN_ID' 或其他
if id_field not in basins.columns:
    raise ValueError(f"流域数据中未找到字段 '{id_field}'，请检查字段名。")

# 为每条河流分配流域（基于相交）
print("正在为河流分配流域...")
river_basin_matches = gpd.sjoin(rivers, basins[[id_field, 'geometry']], how="left", predicate="intersects")
        
# 按流域 ID 分组，生成每个流域的河流集合
print("正在生成每个流域的河流集合...")
for basin_id in basins[id_field].unique():
    # 获取当前流域的河流
    basin_rivers = river_basin_matches[river_basin_matches[id_field] == basin_id]
    
    # 如果该流域没有河流，跳过
    if basin_rivers.empty:
        print(f"流域 {basin_id} 没有匹配的河流，跳过。")
        continue
    
    # 提取原始河流数据的子集（保留所有属性）
    basin_rivers_gdf = rivers.loc[basin_rivers.index].copy()
    
    # 构造输出文件名（确保文件名合法）
    safe_basin_id = str(basin_id).replace("/", "_").replace("\\", "_")  # 替换非法字符
    output_path = os.path.join(output_dir, f"{safe_basin_id}.gpkg")
    
    # 保存为 GeoPackage
    try:
        basin_rivers_gdf.to_file(output_path, driver="GPKG")
        print(f"已保存流域 {basin_id} 的河流到 {output_path}")
    except Exception as e:
        print(f"保存流域 {basin_id} 的河流失败：{e}")

print("所有流域的河流集合已生成！")