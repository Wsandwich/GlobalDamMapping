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
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib
from simplification.cutil import simplify_coords


matplotlib.use('agg')  # Set at the beginning of your script

# Define base directory
base_dir = r"/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/Controlled_experiment/GDAT_dof_result"
output_path = os.path.join(base_dir, "output", "draw")
os.makedirs(output_path, exist_ok=True)

# Find all gpkg files
gpkg_files = glob.glob(os.path.join(base_dir, "*.gpkg"))
gpkg_files = gpkg_files[:20]
print(f"Found {len(gpkg_files)} GPKG files")


# gpkg_files = ['/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/Controlled_experiment/GDAT_dof_result/7030047060.gpkg']

gpkg_files = ['/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/Controlled_experiment/GDAT_dof_result/7030042040.gpkg']

# 蓝色渐变：从浅蓝到深蓝，用于 DOF=0 的河流，基于 RIV_ORD
blue_colors = ['#21356A', '#2271B3', '#9BC8DF']  # 深蓝 -> 中蓝 -> 浅蓝
blue_cmap = LinearSegmentedColormap.from_list("blue_cmap", blue_colors, N=100)
blue_norm = Normalize(vmin=1, vmax=7)  # RIV_ORD 通常从 1 到 7

# 黄色到红色渐变：用于 dof_all 非零的河流
yellow_red_colors = ['#FFFFE0', '#FFA475', '#DB4551', '#8B0100']
yellow_red_cmap = LinearSegmentedColormap.from_list("yellow_red_cmap", yellow_red_colors, N=100)
yellow_red_norm = Normalize(vmin=0, vmax=100)  # dof_all 的值范围


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

# Create lists to store GeoDataFrames
all_dof_gdf_list = []
all_dof_all_gdf_list = []

# Read and process all files
for gpkg_file in gpkg_files:
    # 列出所有图层
    layers = fiona.listlayers(gpkg_file)
    print("GPKG 文件中的所有图层：", layers)

    # 检查每个图层的几何类型
    for layer_name in layers:
        with fiona.open(gpkg_file, layer=layer_name) as layer:
            print(f"\n图层 {layer_name} 的几何类型：")
            print(layer.schema['geometry'])

    file_name = os.path.basename(gpkg_file)
    print(f"Processing file: {file_name}")
    
    try:
        # Try to read DOF layer
        try:
            # 计时：读取 DOF 图层
            start_time = time.time()
            dof_gdf = gpd.read_file(gpkg_file, layer='DOF', usecols=['geometry', 'DOF', 'DIS_AV_CMS', 'dof_all', 'RIV_ORD'])
            elapsed_time = time.time() - start_time
            print(f"  - DOF layer records: {len(dof_gdf)} (took {elapsed_time:.2f} seconds)")
            
            # Process DOF field
            if 'DOF' in dof_gdf.columns:
                # 计时：转换 DOF 列
                start_time = time.time()
                dof_gdf['DOF'] = pd.to_numeric(dof_gdf['DOF'], errors='coerce').fillna(0)
                elapsed_time = time.time() - start_time
                print(f"  - Converted DOF column (took {elapsed_time:.2f} seconds)")
                
                # Add line width based on discharge
                if 'RIV_ORD' in dof_gdf.columns:
                    # 计时：应用 width_logic
                    start_time = time.time()
                    dof_gdf = dof_gdf[dof_gdf['RIV_ORD'] <= 7]
                    dof_gdf['width'] = dof_gdf['RIV_ORD'].apply(width_logic)
                    elapsed_time = time.time() - start_time
                    print(f"  - Applied width_logic to RIV_ORD (took {elapsed_time:.2f} seconds)")
                else:
                    dof_gdf['width'] = 0.5  # Default width
                    print(f"  - Set default width 0.5 (no RIV_ORD column)")
                
                # 计时：过滤非 MultiLineString 几何
                start_time = time.time()
                valid_dof_gdf = dof_gdf[dof_gdf.geometry.apply(lambda x: isinstance(x, MultiLineString))]
                elapsed_time = time.time() - start_time
                if len(valid_dof_gdf) < len(dof_gdf):
                    print(f"  - Warning: Removed {len(dof_gdf) - len(valid_dof_gdf)} non-LineString geometries (took {elapsed_time:.2f} seconds)")
                else:
                    print(f"  - No non-MultiLineString geometries removed (took {elapsed_time:.2f} seconds)")
                
                # Ensure the GeoDataFrame has a CRS
                if valid_dof_gdf.crs is None:
                    start_time = time.time()
                    valid_dof_gdf.crs = "EPSG:4326"  # Set to WGS 84 if missing
                    elapsed_time = time.time() - start_time
                    print(f"  - Set CRS to EPSG:4326 (took {elapsed_time:.2f} seconds)")
                
                all_dof_gdf_list.append(valid_dof_gdf)
                print(f"  - Added valid_dof_gdf to list")
        except Exception as e:
            print(f"Error processing file {file_name}: {str(e)}")
        
        # Process dof_all field - check if in DOF layer
        if 'dof_all' in dof_gdf.columns:
            dof_gdf_copy = dof_gdf.copy()
            dof_gdf_copy['dof_all'] = pd.to_numeric(dof_gdf_copy['dof_all'], errors='coerce').fillna(0)
            
            # Filter out non-LineString geometries
            valid_dof_all_gdf = dof_gdf_copy[dof_gdf_copy.geometry.apply(lambda x: isinstance(x, MultiLineString))]
            
            # Ensure the GeoDataFrame has a CRS
            if valid_dof_all_gdf.crs is None:
                valid_dof_all_gdf.crs = "EPSG:4326"  # Set to WGS 84 if missing
                
            all_dof_all_gdf_list.append(valid_dof_all_gdf)
        else:
            # Try to find dof_all in other layers
            try:
                import fiona
                layers = fiona.listlayers(gpkg_file)
                
                for layer in layers:
                    if layer == 'DOF':  # Already processed
                        continue
                    
                    try:
                        layer_gdf = gpd.read_file(gpkg_file, layer=layer)
                        if 'dof_all' in layer_gdf.columns:
                            print(f"  - Found dof_all field in layer '{layer}'")
                            layer_gdf['dof_all'] = pd.to_numeric(layer_gdf['dof_all'], errors='coerce').fillna(0)
                            
                            # Add line width
                            if 'RIV_ORD' in layer_gdf.columns:
                                layer_gdf['width'] = layer_gdf['RIV_ORD'].apply(width_logic)
                            else:
                                layer_gdf['width'] = 0.5  # Default width
                            
                            # Filter out non-LineString geometries
                            valid_layer_gdf = layer_gdf[layer_gdf.geometry.apply(lambda x: isinstance(x, LineString))]
                            
                            # Ensure the GeoDataFrame has a CRS
                            if valid_layer_gdf.crs is None:
                                valid_layer_gdf.crs = "EPSG:4326"  # Set to WGS 84 if missing
                                
                            all_dof_all_gdf_list.append(valid_layer_gdf)
                            break
                    except Exception as e:
                        print(f"  - Error reading layer '{layer}': {str(e)}")
            except Exception as e:
                print(f"  - Error listing layers: {str(e)}")
    except Exception as e:
        print(f"Error processing file {file_name}: {str(e)}")

# Check if we have data to plot
if not all_dof_gdf_list:
    print("No DOF data found for plotting")
else:
    print(f"Combining {len(all_dof_gdf_list)} datasets for DOF map")

if not all_dof_all_gdf_list:
    print("No dof_all data found for plotting")
else:
    print(f"Combining {len(all_dof_all_gdf_list)} datasets for dof_all map")

# Download and prepare world boundaries (need internet connection)
try:
    # Download world map for context
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    has_world_map = True
    print("Successfully loaded world map")
except Exception as e:
    # Fallback if naturaleartth_lowres is not available
    print(f"Error loading world map: {e}")
    has_world_map = False
    print("Will proceed without world boundaries")

# 修改后的绘图函数
def plot_rivers_on_world_map(output_file, gdf_list, title, dof_col, hatch_pattern='////////////', hatch_color='#DED9E3', hatch_linewidth=0.4):

    processed_gdfs = []
    for gdf in gdf_list:
            # 过滤小流量河流
        if 'RIV_ORD' in gdf.columns:
            gdf = gdf[gdf['RIV_ORD'] <= 7].copy()
        
        # 简化并投影几何形状
        gdf['projected_geom'] = gdf['geometry'].apply(
            lambda g: simplify_and_project_geometry(g, tolerance=0.01)
        )
        processed_gdfs.append(gdf)

    # Set global hatching properties
    plt.rcParams['hatch.color'] = hatch_color
    plt.rcParams['hatch.linewidth'] = hatch_linewidth


    # 创建带有 Robinson 投影的画布
    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.Robinson()})


    # 添加灰色陆地底图
    if has_world_map:
        land = cfeature.NaturalEarthFeature(
            'physical', 'land', '10m',
            facecolor='#ffffff',
            # edgecolor=hatch_color,  # Set the hatch color here
            hatch=hatch_pattern,  # 从左上到右下的斜线
            # linewidth=hatch_linewidth  # This will affect the hatch linewidth
        )
        ax.add_feature(land)

    else:
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)

    ax.spines[:].set_visible(False)

    # 创建线段集合用于绘制河流
    segments = []
    color_values = []
    widths = []
    start_time = time.time()
    for gdf in processed_gdfs:
        for idx, row in gdf.iterrows():
            try:
                geom = row.geometry
                riv_ord = row['RIV_ORD'] if 'RIV_ORD' in row else 1
                width = row['width']
                value = row[dof_col]

                if value == 0:
                    # 值等于 0 时，根据 RIV_ORD 选择蓝色
                    color_value = blue_norm(riv_ord)
                    cmap = blue_cmap
                else:
                    color_value = yellow_red_norm(value)
                    cmap = yellow_red_cmap

                if isinstance(geom, LineString):
                    coords = list(geom.coords)
                    segments.append(coords)
                    color_values.append(cmap(color_value))  # 直接存储颜色值
                    widths.append(width)
                elif isinstance(geom, MultiLineString):
                    for line in geom.geoms:
                        coords = list(line.coords)
                        segments.append(coords)
                        color_values.append(cmap(color_value))
                        widths.append(width)
            except Exception as e:
                print(f"Error processing geometry at index {idx}: {e}")

    end_time = time.time()
    print(f"  - Created line segments (took {end_time - start_time:.2f} seconds)")

    start_time_LineCollection = time.time()
    simplified_segments = [simplify_coords(seg, epsilon=1) for seg in segments]

    # 绘制河流线段
    if simplified_segments:
        # 使用 PlateCarree 投影转换线段坐标
        start_time = time.time()
        lc = LineCollection(
            simplified_segments,
            linewidths=widths,
            colors=color_values,  # 直接使用颜色值
            transform=ccrs.PlateCarree(),
            rasterized=True,
            capstyle='round',
            joinstyle='round'
        )
        end_time = time.time()
        print(f"  - Created LineCollection (took {end_time - start_time:.2f} seconds)")

        # start_time = time.time()
        # lc.set_array(np.array(color_values))
        # end_time = time.time()
        # print(f"  - Set array for LineCollection (took {end_time - start_time:.2f} seconds)")
        start_time = time.time()
        ax.add_collection(lc)
        end_time = time.time()
        print(f"  - Added LineCollection to plot (took {end_time - start_time:.2f} seconds)")    

    end_time_LineCollection = time.time()
    print('------------- End of plot_rivers_on_world_map function -----------------')
    print(f"  - Total time: {end_time_LineCollection - start_time_LineCollection:.2f} seconds")

    start_time = time.time()
    # 设置地图范围为全球
    ax.set_global()  # 显示完整的全球范围

    # 添加色条
    sm_blue = ScalarMappable(cmap=blue_cmap, norm=blue_norm)
    sm_blue.set_array([])
    cbar_blue = plt.colorbar(sm_blue, ax=ax, orientation='horizontal', fraction=0.04, pad=0.1, location='top')
    cbar_blue.set_label('River Order (RIV_ORD) for Zero Values', fontsize=12)

    # 黄色到红色色条（用于值非零）
    sm_yellow_red = ScalarMappable(cmap=yellow_red_cmap, norm=yellow_red_norm)
    sm_yellow_red.set_array([])
    cbar_yellow_red = plt.colorbar(sm_yellow_red, ax=ax, orientation='horizontal', fraction=0.04, pad=0.15, location='bottom')
    cbar_yellow_red.set_label(f'{dof_col} Score (Non-Zero Values)', fontsize=12)

    # 添加数据源说明
    # plt.figtext(0.5, 0.02, "Data source: River fragmentation analysis", ha='center', fontsize=10)

    end_time = time.time()
    print(f"  - Added colorbar and source info (took {end_time - start_time:.2f} seconds)")

    # 保存图像
    plt.savefig(output_file, dpi=2000, bbox_inches='tight')
    print(f"Map saved to {output_file}")
    plt.close(fig)


# 获取当前时间并格式化为字符串
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式示例：20250419_143022

# Plot DOF map
if all_dof_gdf_list:
    plot_rivers_on_world_map(
        os.path.join(output_path, f"global_DOF_map_{current_time}.png"),
        all_dof_gdf_list,
        "River Fragmentation - DOF Values",
        'DOF'
    )

# Plot dof_all map
if all_dof_all_gdf_list:
    plot_rivers_on_world_map(
        os.path.join(output_path, f"global_dof_all_map_{current_time}.png"),
        all_dof_all_gdf_list,
        "River Fragmentation - dof_all Values",
        'dof_all'
    )


print("Mapping complete!")