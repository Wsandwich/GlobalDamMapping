import geopandas as gpd
import pandas as pd

# 文件路径
basins_shp_path = "/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/BasinATLAS_Data_v10_shp/BasinATLAS_v10_shp/BasinATLAS_v10_lev04.shp"
river_gpkg_path = "/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/FFR_river_network.gdb"

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

# 任务 1：检查河流是否被两个流域分开
print("检查河流是否跨越多个流域...")
# 为每条河流分配流域
river_basin_intersections = gpd.sjoin(rivers, basins, how="left", predicate="intersects")

# 按河流 ID 分组，统计每个河流关联的流域数量
river_basin_counts = river_basin_intersections.groupby(river_basin_intersections.index).size()

# 找出跨越多个流域的河流
cross_basin_rivers = river_basin_counts[river_basin_counts > 1]
if not cross_basin_rivers.empty:
    print(f"发现 {len(cross_basin_rivers)} 条河流跨越多个流域：")
    for river_idx, count in cross_basin_rivers.items():
        river_id = rivers.loc[river_idx, 'id'] if 'id' in rivers.columns else river_idx
        intersecting_basins = river_basin_intersections.loc[river_idx, 'BASIN_ID'] if 'BASIN_ID' in river_basin_intersections.columns else "未知流域"
        print(f"河流索引 {river_id} 跨越 {count} 个流域，流域 ID: {intersecting_basins.tolist()}")
else:
    print("没有河流跨越多个流域。")

# 任务 2：检查是否所有河流都被某个流域包含
print("检查河流是否完全被某个流域包含...")
# 使用 within 检查每条河流是否完全位于某个流域内
rivers_within_basins = gpd.sjoin(rivers, basins, how="left", predicate="within")

# 查找没有匹配流域的河流（即不完全在任何流域内的河流）
not_within_basins = rivers_within_basins[rivers_within_basins['index_right'].isna()]

if not not_within_basins.empty:
    print(f"发现 {len(not_within_basins)} 条河流未被任何流域完全包含：")
    for idx, row in not_within_basins.iterrows():
        river_id = row['id'] if 'id' in row else idx
        print(f"河流索引 {river_id}")
else:
    print("所有河流都被某个流域完全包含。")

# 可选：保存结果到文件
# rivers_within_basins.to_file("rivers_with_basins.gpkg", driver="GPKG")
# not_within_basins.to_file("rivers_not_in_basins.gpkg", driver="GPKG")