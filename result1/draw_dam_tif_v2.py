import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import rasterio
from rasterio.warp import reproject
import os
from rasterio.enums import Resampling
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter

# 读取流域的Shapefile
shapefile_path = r"D:\修复数据\研究生\project\data\download\mrb_basins.shp"  # 请将路径替换为您的实际文件路径
gdf_all = gpd.read_file(shapefile_path)

# 读取生成的 GeoTIFF 文件
tif_file_path = r'E:\wyj\dam\v4\global_v5_tif\0.05\dams_points_2020_embankment_dam_0.05.tif'  # 替换为您生成的 GeoTIFF 文件路径
with rasterio.open(tif_file_path) as src:
    raster = src.read(1)  # 读取第一个波段
    transform = src.transform
    crs = src.crs
    # 获取栅格的范围
    minx, miny, maxx, maxy = src.bounds
    original_resolution = transform.a  # 假设像元大小在 x 和 y 方向上相同

file_basename = os.path.basename(tif_file_path)

# 应用高斯平滑滤波（可选）
smoothed_raster = gaussian_filter(raster, sigma=1)

# 创建掩膜数组：水坝数量为0的像元将被掩膜
masked_raster = np.ma.masked_where(raster == 0, raster)

# 压缩数组，移除掩膜的像元
compressed_raster = masked_raster.compressed()

# 计算百分位数
max_threshold = np.percentile(compressed_raster, 95)

# 裁剪数据
clipped_raster = np.clip(masked_raster, 0, max_threshold)

# 定义自定义颜色（RGB），并归一化到 [0, 1]
colors = [
    (232/255, 219/255, 115/255),  # 低值颜色
    (241/255, 167/255, 64/255),   # 中值颜色
    (218/255, 93/255, 49/255),    # 高值颜色
    (139/255, 81/255, 77/255)     # 额外颜色
]


# 创建 ListedColormap
cmap = ListedColormap(colors)

# 定义类别边界，根据数据范围将数据分为三个类别
num_classes = 4
min_val = clipped_raster.min()
max_val = clipped_raster.max()
bounds = np.linspace(min_val, max_val, num_classes + 1)
norm = BoundaryNorm(bounds, cmap.N)

# 计算每个类别的中心位置作为颜色条的刻度位置
tick_positions = (bounds[:-1] + bounds[1:]) / 2
tick_labels = [f"{bounds[i]:.2f} - {bounds[i+1]:.2f}" for i in range(num_classes)]

# 生成经纬度坐标（反转 Y 方向）
new_width = clipped_raster.shape[1]
new_height = clipped_raster.shape[0]
x = np.linspace(minx, maxx, new_width)
y = np.linspace(maxy, miny, new_height)  # 注意这里不再反转 Y 方向，因为 rasterio 的 from_origin 已经处理了
X, Y = np.meshgrid(x, y)


ax = plt.axes(projection=ccrs.Robinson())

ax.set_global()
# 设置地图显示的区域（避免南极）
# ax.set_extent([-180, 180, -90, 90])  # 设置纬度范围在 -60 到 90 之间
# 绘制栅格数据，使用自定义的颜色映射和归一化
im = ax.pcolormesh(X, Y, clipped_raster, cmap=cmap, norm=norm, shading='auto', transform=ccrs.PlateCarree(), zorder=10)

# 232,219,115；241，167,64；218,93,49；139，81,77
# 添加底图细节
land = cfeature.NaturalEarthFeature('physical', 'land', '10m', facecolor=[78/255, 78/255, 78/255])
ax.add_feature(land)

# ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
# ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS, linewidth=0.2, edgecolor=[151/255, 182/255, 225/255])


# 将 GeoDataFrame 的边界提取为坐标并绘制为白色虚线
for _, row in gdf_all.iterrows():
    if row.geometry.geom_type == 'Polygon':  # 使用 geom_type 代替 type
        x, y = row.geometry.exterior.xy
        ax.plot(x, y, transform=ccrs.PlateCarree(), color='white', linestyle='--', linewidth=0.05, zorder=20)
    elif row.geometry.geom_type == 'MultiPolygon':  # 使用 geom_type 代替 type
        for poly in row.geometry.geoms:  # 使用 .geoms 属性访问 MultiPolygon 中的每个 Polygon
            x, y = poly.exterior.xy
            ax.plot(x, y, transform=ccrs.PlateCarree(), color='white', linestyle='--', linewidth=0.05, zorder=20)


# 添加颜色条，设置刻度位置和标签
# cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.5, pad=0.05, boundaries=bounds, ticks=tick_positions)
# cbar.set_label('Number of Dams', fontsize=12)
# cbar.set_ticklabels(tick_labels)

# 添加标题
# plt.title('Global Distribution Map of Total Dams by Raster', fontsize=15)

# 去除图框（可选）
for spine in ax.spines.values():
    spine.set_visible(False)

# 如果需要叠加矢量数据（如流域边界）
# gdf_all.boundary.plot(ax=ax, edgecolor='black', linewidth=0.2, transform=ccrs.PlateCarree())
# plt.show()
plt.savefig(os.path.join(r'E:\wyj\dam\v4\image\result1', f'{file_basename[:-4]}.png'), dpi=2000, bbox_inches='tight')

