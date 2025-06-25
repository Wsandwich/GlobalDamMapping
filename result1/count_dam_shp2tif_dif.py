import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling

# 文件路径
file_2010 = r"E:\wyj\dam\v4\global_v5_tif\dams_points_2010_0.5.tif"
file_2020 = r"E:\wyj\dam\v4\global_v5_tif\dams_points_2020_0.5.tif"
output_file = r"E:\wyj\dam\v4\global_v5_tif\dams_change_2010_2020_0.5.tif"

# 读取2010年的TIF文件
with rasterio.open(file_2010) as src_2010:
    data_2010 = src_2010.read(1)  # 读取第一个波段
    profile_2010 = src_2010.profile  # 获取元数据
    transform_2010 = src_2010.transform
    nodata_2010 = src_2010.nodata if src_2010.nodata is not None else -9999

# 读取2020年的TIF文件
with rasterio.open(file_2020) as src_2020:
    data_2020 = src_2020.read(1)  # 读取第一个波段
    transform_2020 = src_2020.transform
    nodata_2020 = src_2020.nodata if src_2020.nodata is not None else -9999

# 创建一个与2010年栅格相同形状的数组，用于存储重投影后的2020年数据
data_2020_reprojected = np.full_like(data_2010, fill_value=0, dtype=np.float32)

# 重投影2020年数据到2010年的坐标系和分辨率
reproject(
    source=data_2020,
    destination=data_2020_reprojected,
    src_transform=transform_2020,
    src_crs=src_2020.crs,
    dst_transform=transform_2010,
    dst_crs=src_2010.crs,
    resampling=Resampling.nearest  # 使用最近邻重采样
)

# 将NoData区域设置为0
data_2010 = np.where(data_2010 == nodata_2010, 0, data_2010)
data_2020_reprojected = np.where(data_2020_reprojected == nodata_2020, 0, data_2020_reprojected)

# 计算变化 (2020 - 2010)
change = data_2020_reprojected - data_2010

# 更新输出文件的元数据
profile_2010.update(
    dtype=rasterio.float32,  # 输出为浮点数
    nodata=0  # 将NoData值设置为0
)

# 保存结果到新的TIF文件
with rasterio.open(output_file, 'w', **profile_2010) as dst:
    dst.write(change.astype(rasterio.float32), 1)

print(f"Change TIF file saved as {output_file}")