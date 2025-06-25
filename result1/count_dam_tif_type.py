import os
import geopandas as gpd
import rasterio
from shapely.geometry import Polygon, Point
import numpy as np
from tqdm import tqdm
from multiprocessing import cpu_count, Pool
import logging
from rasterio.transform import from_origin

# 定义要处理的年份列表
global years
years = [2010, 2015, 2020]

# def setup_logging(log_file='rasterize_dams_points.log'):
#     """
#     设置日志记录。
#     """
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_file),
#             logging.StreamHandler()
#         ]
#     )

def setup_logging():
    """
    设置日志记录，只输出到终端。
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def check_file_structure(file_path, num_lines=5):
    """
    检查文本文件的结构，打印前几行以帮助确定水坝类型字段的位置。
    
    参数:
        file_path (str): 文件路径
        num_lines (int): 要检查的行数
    """
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                parts = line.strip().split()
                print(f"行 {i+1}: {line.strip()}")
                logging.info(f"行 {i+1} 包含 {len(parts)} 个字段:")
                for j, part in enumerate(parts):
                    logging.info(f"  字段 {j+1}: {part}")
    except Exception as e:
        logging.error(f"检查文件结构时出错: {e}")


def parse_line_to_point(line, dam_type_index=9, dam_types=None):
    """
    解析文本行并转换为 Point 对象（多边形的质心），可以筛选特定类型的水坝。

    参数:
        line (str): 一行文本，包含坐标和其他信息。
        dam_type_index (int): 水坝类型字段在行中的索引位置（从0开始）。
        dam_types (list): 要筛选的水坝类型列表，如果为 None 则包含所有类型。

    返回:
        tuple: (Point, dam_type) 解析得到的点几何对象和水坝类型，或 (None, None) 如果解析失败或类型不匹配。
    """
    try:
        parts = line.strip().split()
        if len(parts) <= dam_type_index:
            return None, None  # 行中的字段不足
        
        # 提取水坝类型
        dam_type = parts[dam_type_index]
        
        if dam_type == 'other_dam':
            dam_type = 'Barrage_dam'
            
        # 如果指定了要筛选的水坝类型，检查当前水坝是否属于这些类型
        if dam_types and dam_type not in dam_types:
            return None, None  # 不是要筛选的水坝类型
            
        # 提取前8个元素作为坐标
        if len(parts) < 8:
            return None, None  # 坐标不足以形成多边形
            
        coords = list(map(float, parts[:8]))
        # 形成 (x, y) 对
        points = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
        # 创建多边形并计算质心
        polygon = Polygon(points)
        if not polygon.is_valid:
            return None, None
        centroid = polygon.centroid
        return centroid, dam_type
    except Exception as e:
        logging.error(f"解析行时出错: {line}\n错误: {e}")
        return None, None


def read_merged_txt_to_gdf_points(file_path, year, dam_types=None, dam_type_index=9):
    """
    从合并后的文本文件读取数据并转换为包含点几何的 GeoDataFrame，可以筛选特定类型的水坝。

    参数:
        file_path (str): 合并后的文本文件路径。
        year (int): 年份。
        dam_types (list): 要筛选的水坝类型列表，如果为 None 则包含所有类型。
        dam_type_index (int): 水坝类型字段在行中的索引位置（从0开始）。

    返回:
        GeoDataFrame: 包含点几何和属性的 GeoDataFrame。
    """
    geometries = []
    dam_type_values = []
    
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc=f"解析 {year} 年的数据" + (f"（筛选类型: {dam_types}）" if dam_types else ""), unit="行"):
            point, dam_type = parse_line_to_point(line, dam_type_index, dam_types)
            if point:
                geometries.append(point)
                dam_type_values.append(dam_type)
    
    if not geometries:
        logging.warning(f"未找到符合条件的水坝数据（年份：{year}，类型：{dam_types}）")
        return gpd.GeoDataFrame()
        
    gdf = gpd.GeoDataFrame({
        'geometry': geometries,
        'dam_type': dam_type_values,
        'year': year
    }, crs='EPSG:4326')  # 输入坐标为 WGS84，经度和纬度

    logging.info(f"已加载 {len(gdf)} 条水坝数据（年份：{year}，类型：{dam_types if dam_types else '全部'}）")
    
    # 可以统计每种类型的数量
    if not gdf.empty:
        type_counts = gdf['dam_type'].value_counts()
        logging.info(f"水坝类型统计：\n{type_counts}")
    
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
        args (tuple): 包含 (year, base_input_dir, base_output_dir, resolution, dam_types, dam_type_index) 的元组。
    """
    year, base_input_dir, base_output_dir, resolution, dam_types, dam_type_index = args
    try:
        input_file = os.path.join(base_input_dir, f"all_{year}_geo_0.4_filtered_all_merged.txt")
        if not os.path.exists(input_file):
            logging.warning(f"输入文件不存在: {input_file}")
            return

        # 如果是第一年且未提供水坝类型索引，检查文件结构以帮助确定水坝类型字段的位置
        if year == min(years) and dam_type_index is None:
            logging.info(f"检查文件结构以确定水坝类型字段的位置: {input_file}")
            check_file_structure(input_file)
            return  # 在确定字段位置后手动设置 dam_type_index 并重新运行
            
        # 处理特定类型的水坝
        if dam_types:
            for dam_type in dam_types:
                logging.info(f"开始处理年份 {year} 的 {dam_type} 类型水坝: {input_file}")
                gdf = read_merged_txt_to_gdf_points(input_file, year, [dam_type], dam_type_index)
                
                if gdf.empty:
                    logging.warning(f"年份 {year} 没有 {dam_type} 类型的水坝")
                    continue
                    
                # 定义输出文件路径，包含水坝类型信息
                output_file = os.path.join(base_output_dir, f"dams_points_{year}_{dam_type}_{resolution}.tif")
                
                rasterize_points_gdf(gdf, output_file, resolution)
                
            # 生成所有指定类型的组合栅格
            logging.info(f"开始处理年份 {year} 的所有指定类型水坝: {input_file}")
            gdf = read_merged_txt_to_gdf_points(input_file, year, dam_types, dam_type_index)
            
            if not gdf.empty:
                # 定义所有类型组合的输出文件
                output_file = os.path.join(base_output_dir, f"dams_points_{year}_all_specified_types_{resolution}.tif")
                rasterize_points_gdf(gdf, output_file, resolution)
        else:
            # 处理所有类型的水坝
            logging.info(f"开始处理年份 {year} 的所有类型水坝: {input_file}")
            gdf = read_merged_txt_to_gdf_points(input_file, year, None, dam_type_index)
            
            # 定义输出文件路径
            output_file = os.path.join(base_output_dir, f"dams_points_{year}_{resolution}.tif")
            
            rasterize_points_gdf(gdf, output_file, resolution)
            
    except Exception as e:
        logging.error(f"处理年份 {year} 时发生错误: {e}")


def main():
    setup_logging()

    # 定义栅格分辨率（单位：度）
    resolution = 1.0

    # 定义基础输入和输出目录
    base_input_dir = r'E:/wyj/dam/v4/global_v5'  # 替换为您的合并文本文件所在目录
    base_output_dir = f'E:/wyj/dam/v4/global_v5_tif/{resolution}'  # 修改输出目录
    os.makedirs(base_output_dir, exist_ok=True)

    
    # 定义要筛选的水坝类型
    dam_types = ['gravity_dam', 'Barrage_dam', 'other_dam', 'arch_dam', 'embankment_dam']
    
    # 定义水坝类型字段在行中的索引位置（从0开始）
    # 如果不确定，先设置为 None，程序会打印文件结构帮助确定
    dam_type_index = 8  # 假设水坝类型在第9个位置（索引为8）
    

    # 如果不确定字段位置，先运行一次检查文件结构
    first_file = os.path.join(base_input_dir, f"all_{min(years)}_geo_0.4_filtered_all_merged.txt")
    if os.path.exists(first_file) and dam_type_index is None:
        logging.info(f"检查文件结构以确定水坝类型字段的位置: {first_file}")
        check_file_structure(first_file)
        logging.info("请根据上述输出确定水坝类型字段的位置，然后设置 dam_type_index 并重新运行程序。")
        return

    # 创建任务列表
    tasks = [(year, base_input_dir, base_output_dir, resolution, dam_types, dam_type_index) for year in years]

    # 使用多进程处理
    num_workers = min(cpu_count(), len(tasks))  # 限制进程数不超过任务数
    with Pool(processes=num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_year_points, tasks), total=len(tasks), desc="总进度", unit="年份"):
            pass

    logging.info("所有年份的点栅格化处理完毕。")


if __name__ == "__main__":
    main()