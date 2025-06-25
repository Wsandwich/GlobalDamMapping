import geopandas as gpd
import os
import pandas as pd
from pathlib import Path
import fiona
from tqdm import tqdm


# 文件路径
basins_shp_path = "/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/BasinATLAS_Data_v10_shp/BasinATLAS_v10_shp/BasinATLAS_v10_lev03.shp"
dams_shp_path = "/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/wyj/大坝知识图谱材料/大坝知识图谱材料/代码/detection/mmrotate-main/out_atlas/global_v5_shp/dams_2020.shp"  # 请替换为实际水坝 shapefile 路径

# 输出目录
output_dir = "/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/result3/dams_by_basin_2010"

os.makedirs(output_dir, exist_ok=True)

# 加载流域和水坝数据
print("正在加载流域数据...")
basins = gpd.read_file(basins_shp_path)
print(basins.head())


# 检查文件是否存在
if not os.path.exists(dams_shp_path):
    raise FileNotFoundError(f"文件未找到：{dams_shp_path}")

# 使用 fiona 打开 shapefile，获取要素总数
with fiona.open(dams_shp_path, 'r') as src:
    total_features = len(src)
    print(f"总共有 {total_features} 个水坝要素")

    # 使用 tqdm 显示进度条，逐条读取
    features = []
    for feature in tqdm(src, total=total_features, desc="加载水坝数据"):
        features.append(feature)

    
# 将 fiona 读取的要素转换为 GeoDataFrame
dams = gpd.GeoDataFrame.from_features(features, crs='EPSG:4326')
print("水坝数据加载完成！")


# dams = gpd.read_file(dams_shp_path)

# # 检查水坝数据的 CRS
# if dams.crs is None:
#     print("水坝数据没有定义 CRS，假设为 EPSG:4326（请确认实际 CRS）...")
#     dams.set_crs(epsg=4326, inplace=True)  # 假设水坝数据为 EPSG:4326，需确认
# else:
#     print(f"水坝 CRS: {dams.crs}")


# 检查并统一坐标参考系统（CRS）
if basins.crs != dams.crs:
    print(f"流域 CRS: {basins.crs}, 水坝 CRS: {dams.crs}")
    print("正在将水坝数据转换为流域的 CRS...")
    dams = dams.to_crs(basins.crs)

# 确定流域的唯一标识字段（请根据实际数据调整）
# 假设流域有一个字段如 'HYBAS_ID'（BasinATLAS 常用）
id_field = 'HYBAS_ID'  # 请替换为实际字段名，例如 'BASIN_ID' 或其他
if id_field not in basins.columns:
    raise ValueError(f"流域数据中未找到字段 '{id_field}'，请检查字段名。")

# 为每个水坝点分配流域（基于包含关系）
print("正在为水坝分配流域...")
dam_basin_matches = gpd.sjoin(dams, basins[[id_field, 'geometry']], how="left", predicate="within")

# 按流域 ID 分组，生成每个流域的水坝集合
print("正在生成每个流域的水坝集合...")
for basin_id in basins[id_field].unique():
    # 获取当前流域的水坝
    basin_dams = dam_basin_matches[dam_basin_matches[id_field] == basin_id]
    
    # 如果该流域没有水坝，跳过
    if basin_dams.empty:
        print(f"流域 {basin_id} 没有匹配的水坝，跳过。")
        continue
    
    # 提取原始水坝数据的子集（保留所有属性）
    basin_dams_gdf = dams.loc[basin_dams.index].copy()
    
    # 构造输出文件名（确保文件名合法）
    safe_basin_id = str(basin_id).replace("/", "_").replace("\\", "_")  # 替换非法字符
    output_path = os.path.join(output_dir, f"{safe_basin_id}.shp")
    
    # 保存为 shapefile
    try:
        basin_dams_gdf.to_file(output_path, driver="ESRI Shapefile")
        print(f"已保存流域 {basin_id} 的水坝到 {output_path}")
    except Exception as e:
        print(f"保存流域 {basin_id} 的水坝失败：{e}")

print("所有流域的水坝集合已生成！")