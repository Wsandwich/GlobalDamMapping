import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import rasterio
import pandas as pd
import os
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Define the years to process
years = [2010, 2015, 2020]

# Define color scheme
colors = ['#FEDF73', '#FACBC8', '#A73325']

# Define GeoTIFF file path template
tif_file_template = 'E:/wyj/dam/v4/global_v5_tif/dams_points_{year}_0.05.tif'

# Create dictionary to store data for each year
data_per_year = {}

# Define latitude band width
latitude_band_width = 0.05

# Process data for each year
for idx, year in enumerate(years):
    tif_file_path = tif_file_template.format(year=year)
    if not os.path.exists(tif_file_path):
        print(f"File does not exist: {tif_file_path}")
        continue
    
    with rasterio.open(tif_file_path) as src:
        raster = src.read(1)  # Read first band
        transform = src.transform
        crs = src.crs
        # Get raster bounds
        minx, miny, maxx, maxy = src.bounds
        # Get pixel size
        pixel_size_x = transform.a
        pixel_size_y = -transform.e  # transform.e is usually negative

    # Generate lat/lon coordinates
    new_width = raster.shape[1]
    new_height = raster.shape[0]
    x = np.linspace(minx, maxx, new_width)
    y = np.linspace(maxy, miny, new_height)
    X, Y = np.meshgrid(x, y)

    # Create DataFrame to store latitude and dam count for each pixel
    latitudes = Y.flatten()
    dam_counts = raster.flatten()

    df = pd.DataFrame({
        'latitude': latitudes,
        'dam_count': dam_counts
    })

    # Remove pixels with zero dams
    df = df[df['dam_count'] > 0]

    # Filter out values outside latitude range
    df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90)]

    # Generate bins
    bins = np.arange(-90, 90 + latitude_band_width, latitude_band_width)
    # Generate labels using the lower bound of each band, ensuring float precision
    labels = np.round(bins[:-1], decimals=5)

    # Create latitude bands using pandas.cut
    df['lat_band'] = pd.cut(
        df['latitude'],
        bins=bins,  # From -90 to 90 degrees, each band is latitude_band_width wide
        right=False,  # Left-closed, right-open interval [a, b)
        include_lowest=True,
        labels=labels  # Use the lower bound of each band as the label
    )

    # Verify labels and bins match in length
    assert len(labels) == len(bins) - 1, "Number of labels must be one less than number of bins"

    # Sum dam counts in each latitude band
    count_per_lat_band = df.groupby('lat_band')['dam_count'].sum().reset_index()

    # Convert lat_band to numeric type
    count_per_lat_band['lat_band'] = count_per_lat_band['lat_band'].astype(float)

    # Ensure latitude bands are sorted
    count_per_lat_band = count_per_lat_band.sort_values('lat_band')

    # Store in dictionary
    data_per_year[year] = count_per_lat_band

# Merge data from all years
merged_df = pd.DataFrame({'lat_band': data_per_year[years[0]]['lat_band']})

for year in years:
    if year in data_per_year:
        merged_df = merged_df.merge(
            data_per_year[year][['lat_band', 'dam_count']], 
            on='lat_band', 
            how='left', 
            suffixes=('', f'_{year}')
        )
        merged_df.rename(columns={'dam_count': f'dam_count_{year}'}, inplace=True)
    else:
        merged_df[f'dam_count_{year}'] = 0  # Fill with zeros if data missing for a year

# Fill NA values with 0
for year in years:
    merged_df[f'dam_count_{year}'] = merged_df[f'dam_count_{year}'].fillna(0)

# Define output directory
output_dir = r'E:\wyj\dam\v4\image\result1'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. AREA CHART WITH SIDE-BY-SIDE LAYERS (2020 on top, 2015 in middle, 2010 on bottom)
plt.figure(figsize=(38, 5.5))

# Plot each year as a separate area (not stacked)
for i, year in enumerate(years):
    print(f"Plotting {year}..., zorder={3-i}")
    plt.fill_between(
        merged_df['lat_band'], 
        merged_df[f'dam_count_{year}'],
        alpha=0.95,
        label=str(year),
        color=colors[i],
        zorder=3-i,  # Higher zorder means the layer is placed on top
        edgecolor='none'  # Remove the borders by setting edgecolor to 'none'
    )

# 创建主y轴的引用
ax1 = plt.gca()

# Add legend, title and labels
# ax1.legend(title='Year', loc='upper right', fontsize=12)
# ax1.set_xlabel('Latitude (°)', fontsize=14)
# ax1.set_ylabel('Number of Dams', fontsize=14)
# ax1.set_title('Dam Distribution by Latitude - Area Chart with 2020 (top), 2015 (middle), 2010 (bottom)', fontsize=16)

# 设置x轴范围从-50到75
ax1.set_xlim(-50, 75)

# 设置x轴刻度
selected_ticks = [-50, -25, 0, 25, 50, 75]
ax1.set_xticks(selected_ticks)

# 设置y轴从0开始
ax1.set_ylim(bottom=0)

# Add reference lines
for tick in selected_ticks[1:-1]:
    ax1.axvline(tick, color='gray', linestyle='--', linewidth=3.0, alpha=0.8, zorder=13)

ax1.set_yticks([0, 5000, 10000, 15000, 20000])
ax1.set_ylim(bottom=0, top=26000)
# 创建右侧的副y轴
ax2 = ax1.twinx()
# ax2.set_ylabel('Total Dams in Latitude Range', fontsize=14)

ax2.yaxis.set_visible(False)  # 添加这行来隐藏整个y轴（刻度和标签）

# 计算每个纬度区间内每年的水坝总和
latitude_ranges = [(selected_ticks[i], selected_ticks[i+1]) for i in range(len(selected_ticks)-1)]

# 为每个年份和每个纬度区间计算水坝总数
range_totals = []
for start, end in latitude_ranges:
    year_totals = []
    for year in years:
        # 筛选出在该纬度区间内的数据
        mask = (merged_df['lat_band'] >= start) & (merged_df['lat_band'] < end)
        total = merged_df.loc[mask, f'dam_count_{year}'].sum()
        year_totals.append(total)
    range_totals.append(year_totals)

# 计算最大的总数，以设置副y轴的范围
max_total = max([max(totals) for totals in range_totals]) if range_totals else 0
ax2.set_ylim(0, max_total * 1.1)  # 设置副y轴范围，留出10%的空间


# 在每个区间中点绘制点
for i, ((start, end), totals) in enumerate(zip(latitude_ranges, range_totals)):
    mid_point = (start + end) / 2
    for j, (year, total) in enumerate(zip(years, totals)):
        # 计算x位置，稍微错开以便于查看
        offset = j - 1  # -1, 0, 1
        x_pos = mid_point + offset * (end - start) / 3.5
        
        # 绘制点，使用与面积图相同的颜色
        ax2.scatter(
            x_pos, 
            total, 
            color=colors[j],           # 填充颜色
            s=450,                     # 点的大小，更大的值
            marker='o',                # 点的形状：'o'圆形，'s'方形，'^'三角形，'*'星形等
            edgecolor='black',         # 边框颜色
            linewidth=2.5,             # 边框宽度
            alpha=0.8,                 # 透明度
            zorder=10,
            label=f"{year} Total" if i == 0 else ""
        )
        
        # 在点上方标注值（除以10^6并保留两位小数）
        value_in_millions = total / 1e6

        # 在点上方标注值
        ax2.annotate(f"{value_in_millions:.2f}", 
                    (x_pos, total), 
                    textcoords="offset points",
                    xytext=(0, 25),
                    ha='center',
                    fontsize=40,
                    # fontweight='bold'
                    )  # 加粗文本使其更易读

# 添加副y轴的图例（仅在第一个区间添加一次标签）
handles, labels = ax2.get_legend_handles_labels()
# if handles:
#     ax2.legend(handles, labels, loc='upper left', fontsize=12)

ax1.spines['top'].set_visible(False)  # 隐藏上边框
ax2.spines['top'].set_visible(False)  # 同时也隐藏副y轴的上边框
ax1.spines['right'].set_visible(False)  # 隐藏右边框
ax2.spines['right'].set_visible(False)  # 同时也隐藏副y轴的右边框
# 修改边框粗细
ax1.spines['bottom'].set_linewidth(3.0)  # x轴
ax1.spines['left'].set_linewidth(3.0)    # y轴
ax1.tick_params(axis='x', width=3.0)
ax1.tick_params(axis='y', width=3.0)

# 隐藏刻度标签（数字），但保留刻度线
ax1.set_xticklabels([])  # 或者 ax1.xaxis.set_ticklabels([])
ax1.set_yticklabels([])  # 或者 ax1.yaxis.set_ticklabels([])

ax1.tick_params(axis='x', length=10.0)  # 设置x轴刻度线长度
ax1.tick_params(axis='y', length=10.0)  # 设置y轴刻度线长度

plt.tight_layout()
plt.savefig(f'{output_dir}/2_area_chart_layered_with_points_2.png', dpi=600, bbox_inches='tight')
plt.close()


# Print the values of the points
print("\nTotal dam counts by latitude range and year:")
print("="*50)
print(f"{'Latitude Range':<20} {'2010':>10} {'2015':>10} {'2020':>10}")
print("-"*50)
for i, ((start, end), totals) in enumerate(zip(latitude_ranges, range_totals)):
    print(f"{start} to {end:<13} {totals[0]:>10.0f} {totals[1]:>10.0f} {totals[2]:>10.0f}")
print("="*50)

# Print the values in millions (as shown in annotations)
print("\nTotal dam counts in millions:")
print("="*50)
print(f"{'Latitude Range':<20} {'2010':>10} {'2015':>10} {'2020':>10}")
print("-"*50)
for i, ((start, end), totals) in enumerate(zip(latitude_ranges, range_totals)):
    values_in_millions = [total / 1e6 for total in totals]
    print(f"{start} to {end:<13} {values_in_millions[0]:>10.2f} {values_in_millions[1]:>10.2f} {values_in_millions[2]:>10.2f}")
print("="*50)