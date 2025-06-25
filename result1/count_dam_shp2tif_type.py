import os
import geopandas as gpd
import rasterio
import numpy as np
from tqdm import tqdm
from multiprocessing import cpu_count, Pool
import logging
from rasterio.transform import from_origin


def setup_logging(log_file='rasterize_dams_shp.log'):
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

def read_shp_to_gdf(file_path, year, dam_types=None):
    """
    从 shapefile 文件读取数据并转换为 GeoDataFrame，可以筛选特定类型的水坝。

    参数:
        file_path (str): shapefile 文件路径。
        year (int): 年份。
        dam_types (list): 要筛选的水坝类型列表，如果为 None 则加载所有类型。

    返回:
        GeoDataFrame: 包含点几何和属性的 GeoDataFrame。
    """
    try:
        if dam_types:
            # 构建 SQL 查询条件，筛选特定类型的水坝
            # 假设字段名为 'dam_type'，根据实际情况调整
            dam_types_str = "', '".join(dam_types)
            sql_query = f"SELECT * FROM {os.path.basename(file_path).split('.')[0]} WHERE dam_type IN ('{dam_types_str}')"
            
            # 使用 SQL 查询加载特定类型的水坝
            gdf = gpd.read_file(file_path, sql=sql_query)
            logging.info(f"已从 {file_path} 筛选加载 {len(dam_types)} 种类型的水坝，共 {len(gdf)} 条记录")
        else:
            # 加载所有数据
            gdf = gpd.read_file(file_path)
            logging.info(f"已从 {file_path} 加载所有水坝，共 {len(gdf)} 条记录")
        
        # 确保几何类型为点（如果不是点，可能需要处理）
        if not all(gdf.geometry.type == 'Point'):
            logging.warning(f"文件 {file_path} 包含非点几何，正在尝试提取质心...")
            gdf['geometry'] = gdf.geometry.centroid
        
        # 添加年份列
        gdf['year'] = year
        
        # 设置或检查 CRS
        if gdf.crs is None:
            logging.warning(f"文件 {file_path} 未定义 CRS，假定为 EPSG:4326")
            gdf.crs = 'EPSG:4326'
        elif gdf.crs.to_epsg() != 4326:
            logging.info(f"将文件 {file_path} 的 CRS 转换为 EPSG:4326")
            gdf = gdf.to_crs('EPSG:4326')
        
        return gdf
    except Exception as e:
        logging.error(f"读取 shapefile {file_path} 时出错: {e}")
        return gpd.GeoDataFrame()


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
    处理单个年份的 shapefile 文件，生成点栅格 GeoTIFF。

    参数:
        args (tuple): 包含 (year, base_input_dir, base_output_dir, resolution, dam_types) 的元组。
    """
    year, base_input_dir, base_output_dir, resolution, dam_types = args
    try:
        input_file = os.path.join(base_input_dir, f"dams_{year}.shp")
        if not os.path.exists(input_file):
            logging.warning(f"输入文件不存在: {input_file}")
            return

        # 为每种水坝类型生成单独的栅格文件
        for dam_type in dam_types:
            logging.info(f"开始处理年份 {year} 的 {dam_type} 类型水坝: {input_file}")
            gdf = read_shp_to_gdf(input_file, year, [dam_type])
            
            if gdf.empty:
                logging.warning(f"年份 {year} 没有 {dam_type} 类型的水坝")
                continue
                
            # 定义输出文件路径，包含水坝类型信息
            output_file = os.path.join(base_output_dir, f"dams_points_{year}_{dam_type}_{resolution}.tif")
            
            rasterize_points_gdf(gdf, output_file, resolution)
            
        # 此外，也可以生成所有指定类型的组合栅格
        logging.info(f"开始处理年份 {year} 的所有指定类型水坝: {input_file}")
        gdf = read_shp_to_gdf(input_file, year, dam_types)
        
        if not gdf.empty:
            # 定义所有类型组合的输出文件
            output_file = os.path.join(base_output_dir, f"dams_points_{year}_all_specified_types_{resolution}.tif")
            rasterize_points_gdf(gdf, output_file, resolution)
            
    except Exception as e:
        logging.error(f"处理年份 {year} 时发生错误: {e}")

def main():
    setup_logging()

    # 定义基础输入和输出目录
    base_input_dir = r'E:\wyj\dam\v4\global_v5_shp'  # 替换为您的 shapefile 文件所在目录
    base_output_dir = r'E:\wyj\dam\v4\global_v5_tif'  # 修改输出目录
    os.makedirs(base_output_dir, exist_ok=True)

    # 定义要处理的年份列表
    years = [2010, 2015, 2020]

    # 定义栅格分辨率（单位：度）
    resolution = 0.5

    # 定义要筛选的水坝类型
    dam_types = ['gravity_dam', 'Barrage_dam', 'other_dam', 'arch_dam']

    # 创建任务列表
    tasks = [(year, base_input_dir, base_output_dir, resolution, dam_types) for year in years]

    # 使用多进程处理
    num_workers = min(cpu_count(), len(tasks))  # 限制进程数不超过任务数
    with Pool(processes=num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_year_points, tasks), total=len(tasks), desc="总进度", unit="年份"):
            pass

    logging.info("所有年份的点栅格化处理完毕。")


if __name__ == "__main__":
    main()