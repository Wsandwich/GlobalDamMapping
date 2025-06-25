import os
import geopandas as gpd
import rasterio
from shapely.geometry import Polygon, Point
import numpy as np
from tqdm import tqdm
from multiprocessing import cpu_count, Pool
import logging
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_origin


def setup_logging(log_file='rasterize_dams_points.log'):
    """
    设置日志记录。
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def parse_line_to_point(line):
    """
    解析文本行并转换为 Point 对象（多边形的质心）。

    参数:
        line (str): 一行文本，包含坐标和其他信息。

    返回:
        Point: 解析得到的点几何对象，或 None 如果解析失败。
    """
    try:
        parts = line.strip().split()
        if len(parts) < 10:
            return None  # 坐标不足以形成多边形
        # 提取前8个元素作为坐标
        coords = list(map(float, parts[:8]))
        # 形成 (x, y) 对
        points = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
        # 创建多边形并计算质心
        polygon = Polygon(points)
        if not polygon.is_valid:
            return None
        centroid = polygon.centroid
        return centroid
    except Exception as e:
        logging.error(f"解析行时出错: {line}\n错误: {e}")
        return None

def read_merged_txt_to_gdf_points(file_path, year):
    """
    从合并后的文本文件读取数据并转换为包含点几何的 GeoDataFrame。

    参数:
        file_path (str): 合并后的文本文件路径。
        year (int): 年份。

    返回:
        GeoDataFrame: 包含点几何和属性的 GeoDataFrame。
    """
    max_lines = 100000  # 最大解析行数
    line_count = 0  # 初始化计数器
    geometries = []
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc=f"解析 {year} 年的数据", unit="行"):
            if line_count >= max_lines:  # 如果已达到最大行数，停止解析
                break
            point = parse_line_to_point(line)
            if point:
                geometries.append(point)
            # line_count += 1  # 更新计数器
    
    gdf = gpd.GeoDataFrame({
        'geometry': geometries,
        'year': year
    }, crs='EPSG:4326')  # 输入坐标为 WGS84，经度和纬度

    return gdf

def rasterize_points_gdf(gdf, output_file, resolution=0.01):
    """
    将包含点的 GeoDataFrame 栅格化为 GeoTIFF 文件，并统计每个像素中的点数量。
    
    参数:
        gdf (GeoDataFrame): 要栅格化的点 GeoDataFrame。
        output_file (str): 输出 GeoTIFF 文件路径。
        resolution (float): 栅格分辨率（单位：度）。
    """
    try:
        if gdf.empty:
            logging.warning(f"GeoDataFrame 是空的，跳过生成栅格: {output_file}")
            return

        # 检查几何对象是否存在无效坐标
        infinite_geometries = gdf[gdf.geometry.apply(
            lambda geom: not geom.is_valid or
                        any(coord in [np.inf, -np.inf] for coord in geom.bounds))]
        if not infinite_geometries.empty:
            logging.error(f"存在无效或包含无穷大坐标的几何对象，跳过这些几何。数量: {len(infinite_geometries)}")
            gdf = gdf.drop(infinite_geometries.index)
            if gdf.empty:
                logging.warning(f"所有几何对象均无效，跳过生成栅格: {output_file}")
                return

        # 确保 GeoDataFrame 使用 EPSG:4326
        if gdf.crs is None:
            logging.error("GeoDataFrame 没有定义 CRS，请先设置 CRS 为 EPSG:4326。")
            return
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs('EPSG:4326')
            logging.info("已将 GeoDataFrame 的 CRS 转换为 EPSG:4326")

        # 获取边界
        minx, miny, maxx, maxy = gdf.total_bounds

        # 计算栅格尺寸
        width = int(np.ceil((maxx - minx) / resolution))
        height = int(np.ceil((maxy - miny) / resolution))

        # 定义栅格的仿射变换
        transform = from_origin(minx, maxy, resolution, resolution)

        # 创建空栅格
        raster = np.zeros((height, width), dtype=np.uint16)

        # 将每个点映射到栅格并计数
        for point in gdf.geometry:
            if point.is_empty:
                continue
            col, row = ~transform * (point.x, point.y)  # 计算列和行
            col, row = int(col), int(row)
            if 0 <= row < height and 0 <= col < width:
                raster[row, col] += 1

        # 定义栅格的元数据
        meta = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': raster.dtype,
            'crs': 'EPSG:4326',
            'transform': transform
        }

        # 写入 GeoTIFF 文件
        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(raster, 1)

        logging.info(f"已生成 GeoTIFF 文件: {output_file} (投影: EPSG:4326)")
    except Exception as e:
        logging.error(f"栅格化时发生错误: {e}")


def process_year_points(args):
    """
    处理单个年份的合并文本文件，生成点栅格 GeoTIFF。

    参数:
        args (tuple): 包含 (year, base_input_dir, base_output_dir, resolution) 的元组。
    """
    year, base_input_dir, base_output_dir, resolution = args
    try:
        input_file = os.path.join(base_input_dir, f"all_{year}_geo_0.4_filtered_all_merged.txt")
        if not os.path.exists(input_file):
            logging.warning(f"输入文件不存在: {input_file}")
            return

        logging.info(f"开始处理年份 {year} 的文件: {input_file}")
        gdf = read_merged_txt_to_gdf_points(input_file, year)

        # 定义输出文件路径
        output_file = os.path.join(base_output_dir, f"dams_points_{year}_{resolution}.tif")

        rasterize_points_gdf(gdf, output_file, resolution)
    except Exception as e:
        logging.error(f"处理年份 {year} 时发生错误: {e}")

def main():
    setup_logging()

    # 定义基础输入和输出目录
    base_input_dir = r'E:\wyj\dam\v4\global_v5'  # 替换为您的合并文本文件所在目录
    base_output_dir = r'E:\wyj\dam\v4\global_v5_tif'  # 修改输出目录
    os.makedirs(base_output_dir, exist_ok=True)

    # 定义要处理的年份列表
    years = [2010, 2015, 2020]

    # 定义栅格分辨率（单位：米）
    resolution = 0.05

    # 创建任务列表
    tasks = [(year, base_input_dir, base_output_dir, resolution) for year in years]

    # 使用多进程处理
    num_workers = min(cpu_count(), 1)
    with Pool(processes=num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_year_points, tasks), total=len(tasks), desc="总进度", unit="年份"):
            pass

    logging.info("所有年份的点栅格化处理完毕。")

if __name__ == "__main__":
    main()
