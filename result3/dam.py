import geopandas as gpd
from shapely.ops import nearest_points
from shapely.ops import nearest_points
import numpy as np

# 读取数据
streams = gpd.read_file(r"E:\研究生\project\data\free_rivers\LIMPOPO_R.gpkg")
dams = gpd.read_file(r"E:\研究生\project\data\dam_ours\out_shp_v1\2020\LIMPOPO\2020_LIMPOPO_gravity_dam_point.shp")

# 确保两个数据集在同一坐标系统中
dams = dams.to_crs(streams.crs)

# 初始化两个新列以存储关联的GOID和BAS_ID
dams['Nearest_GOID'] = None
dams['Nearest_BAS_ID'] = None

# 计算最近的河段
for index, dam in dams.iterrows():
    # 寻找与当前水坝点最近的河流几何体
    # 这里我们使用 distance 方法来找到实际的最小距离
    nearest_stream = streams.geometry.distance(dam.geometry).idxmin()
    nearest_stream_geom = streams.iloc[nearest_stream]

    # 检查是否找到了对应的河段
    # if not nearest_stream.empty:
    print(f"is {nearest_stream['GOID']}")
    dams.at[index, 'GOID'] = int(nearest_stream['GOID'].values[0])  # 确保为int类型
    dams.at[index, 'BAS_ID'] = int(nearest_stream['BAS_ID'].values[0])  # 确保为int类型

    # else:
        # print(f"No nearby stream found for dam at index {index}")

# 保存结果到新的Shapefile
dams.to_file(r"E:\研究生\project\data\dam_ours\out_shp_v2")