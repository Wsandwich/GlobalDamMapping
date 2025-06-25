import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import rasterio
import pandas as pd
import os

# 定义要处理的年份列表
years = [2010, 2015, 2020]

# 定义配色方案（按照您提供的顺序）
colors = ['#FED59F', '#FDBA83', '#FB8C57']

# 定义GeoTIFF文件路径模板
tif_file_template = 'E:/wyj/dam/v4/global_v5_tif/dams_points_{year}_0.05.tif'

# 定义存储每年汇总数据的字典
data_per_year = {}

# 定义纬度带宽度
latitude_band_width = 0.05  # 可根据需要调整

# 处理每一年的数据
for idx, year in enumerate(years):
    tif_file_path = tif_file_template.format(year=year)
    if not os.path.exists(tif_file_path):
        print(f"文件不存在: {tif_file_path}")
        continue
    
    with rasterio.open(tif_file_path) as src:
        raster = src.read(1)  # 读取第一个波段
        transform = src.transform
        crs = src.crs
        # 获取栅格的范围
        minx, miny, maxx, maxy = src.bounds
        # 获取像元大小
        pixel_size_x = transform.a
        pixel_size_y = -transform.e  # transform.e 通常为负值

    # 生成经纬度坐标
    new_width = raster.shape[1]
    new_height = raster.shape[0]
    x = np.linspace(minx, maxx, new_width)
    y = np.linspace(maxy, miny, new_height)  # rasterio 的 from_origin 已经处理了 Y 方向
    X, Y = np.meshgrid(x, y)

    # 创建一个 DataFrame 来存储每个像元的纬度和水坝数量
    latitudes = Y.flatten()
    dam_counts = raster.flatten()

    df = pd.DataFrame({
        'latitude': latitudes,
        'dam_count': dam_counts
    })

    # 去除水坝数量为0的像元
    df = df[df['dam_count'] > 0]

    # 可选：过滤超出纬度范围的值
    df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90)]

    # 生成 bins
    bins = np.arange(-90, 90 + latitude_band_width, latitude_band_width)
    # 生成 labels，使用带的下限作为标签，并确保浮点数精度
    labels = np.round(bins[:-1], decimals=5)

    # 使用 pandas.cut 创建纬度带
    df['lat_band'] = pd.cut(
        df['latitude'],
        bins=bins,  # 从 -90 到 90 度，每 latitude_band_width 度一个带
        right=False,  # 左闭右开区间 [a, b)
        include_lowest=True,
        labels=labels  # 使用带的下限作为标签
    )

    # 检查 labels 和 bins 的长度是否匹配
    assert len(labels) == len(bins) - 1, "Labels 数量必须比 bins 少 1"

    # 统计每个纬度带内的水坝总数
    count_per_lat_band = df.groupby('lat_band')['dam_count'].sum().reset_index()

    # 将 lat_band 转换为数值类型
    count_per_lat_band['lat_band'] = count_per_lat_band['lat_band'].astype(float)

    # 确保纬度带按顺序排列
    count_per_lat_band = count_per_lat_band.sort_values('lat_band')

    # 存储到字典中
    data_per_year[year] = count_per_lat_band

# 合并三年的数据
merged_df = pd.DataFrame({'lat_band': data_per_year[years[0]]['lat_band']})

for year in years:
    if year in data_per_year:
        merged_df = merged_df.merge(data_per_year[year][['lat_band', 'dam_count']], on='lat_band', how='left', suffixes=('', f'_{year}'))
        merged_df.rename(columns={'dam_count': f'dam_count_{year}'}, inplace=True)
    else:
        merged_df[f'dam_count_{year}'] = 0  # 如果某年数据缺失，填充为0

# 填充缺失值为0
for year in years:
    merged_df[f'dam_count_{year}'] = merged_df[f'dam_count_{year}'].fillna(0)

# 准备绘图数据
x = merged_df['lat_band']
y = [merged_df[f'dam_count_{year}'] for year in years]

# 绘制堆积面积图
plt.figure(figsize=(24, 4))
plt.stackplot(x, y, labels=[str(year) for year in years], colors=colors, alpha=0.8)

# 添加图例
# plt.legend(title='Year', loc='upper left')

# 添加标签和标题
# plt.xlabel('Latitude Band (°)')
# plt.ylabel('Total Number of Dams')
# plt.title(f'Total Number of Dams by Latitude Band (每 {latitude_band_width}°)')

# 设置x轴刻度
selected_ticks = [-50, -25, 0, 25, 50, 75]
plt.xticks(selected_ticks)  # 设置指定的x轴刻度

# 在指定刻度位置绘制虚线
for tick in selected_ticks:
    plt.axvline(tick, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)


# 添加网格
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# 设置 y 轴刻度
plt.yticks([10000, 20000, 30000, 40000])
# 优化布局
plt.tight_layout()


output_file = f'E:/wyj/dam/v4/image/总图/dams_by_latitude_band_stack_{latitude_band_width}deg.png'

if not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))

    
plt.savefig(output_file, dpi=1000, bbox_inches='tight')

