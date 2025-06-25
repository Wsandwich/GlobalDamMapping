import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon, Point
import logging
from tqdm import tqdm
import os

def setup_logging(log_file='basin_dam_statistics.log'):
    """设置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def parse_line_to_point(line, dam_type_index=8):
    """
    解析文本行并转换为 Point 对象（多边形的质心）。

    参数:
        line (str): 一行文本，包含坐标和其他信息。
        dam_type_index (int): 水坝类型字段在行中的索引位置（从0开始）。

    返回:
        tuple: (Point, dam_type) 解析得到的点几何对象和水坝类型，或 (None, None) 如果解析失败。
    """
    try:
        parts = line.strip().split()
        if len(parts) <= dam_type_index:
            return None, None  # 行中的字段不足
        
        # 提取水坝类型
        dam_type = parts[dam_type_index]
            
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

def load_dams_from_txt(file_path, dam_type_index=8):
    """
    从文本文件加载水坝数据并转换为GeoDataFrame。
    
    参数:
        file_path (str): 文本文件路径
        dam_type_index (int): 水坝类型字段在行中的索引位置（从0开始）
        
    返回:
        GeoDataFrame: 包含水坝点位置和类型的GeoDataFrame
    """
    geometries = []
    dam_types = []
    
    logging.info(f"从文件加载水坝数据: {file_path}")
    with open(file_path, 'r') as f:
        for line in tqdm(f, desc="解析水坝数据", unit="行"):
            point, dam_type = parse_line_to_point(line, dam_type_index)
            if point:
                geometries.append(point)
                dam_types.append(dam_type)
    
    if not geometries:
        logging.warning(f"未在文件中找到有效的水坝数据: {file_path}")
        return gpd.GeoDataFrame()
    
    gdf = gpd.GeoDataFrame({
        'geometry': geometries,
        'dam_type': dam_types
    }, crs='EPSG:4326')  # 假设输入坐标为 WGS84
    
    # 统计各类型数量
    type_counts = gdf['dam_type'].value_counts()
    logging.info(f"已加载 {len(gdf)} 个水坝点，类型统计:\n{type_counts}")
    
    return gdf

def count_dams_in_basins(basin_gdf, dams_gdf, year, output_fields):
    """
    统计每个流域中不同类型水坝的数量
    
    参数:
        basin_gdf (GeoDataFrame): 流域GeoDataFrame
        dams_gdf (GeoDataFrame): 水坝GeoDataFrame
        year (int): 年份
        output_fields (dict): 输出字段名映射，如 {'gravity_dam': 'g', 'arch_dam': 'a'}
        
    返回:
        DataFrame: 包含流域ID和水坝计数的DataFrame
    """
    logging.info(f"开始统计 {year} 年流域中的水坝数量")
    
    # 确保两个GeoDataFrame使用相同的坐标系
    if basin_gdf.crs != dams_gdf.crs:
        dams_gdf = dams_gdf.to_crs(basin_gdf.crs)
        logging.info(f"已将水坝数据转换为流域数据的坐标系: {basin_gdf.crs}")
    
    # 空间连接 - 将水坝与其所在的流域关联
    joined = gpd.sjoin(dams_gdf, basin_gdf[['HYBAS_ID', 'geometry']], how='inner', predicate='within')
    logging.info(f"共有 {len(joined)} 个水坝与流域匹配")
    
    # 获取唯一的水坝类型
    dam_types = dams_gdf['dam_type'].unique()
    logging.info(f"水坝类型: {dam_types}")
    
    # 初始化结果DataFrame
    result_data = {'HYBAS_ID': basin_gdf['HYBAS_ID']}
    result_df = pd.DataFrame(result_data)
    
    # 按类型计算每个流域的水坝数量
    for dam_type in dam_types:
        # 使用简写作为字段名
        if dam_type in output_fields:
            field_name = f"{year}_{output_fields[dam_type]}"
            logging.info(f"计算字段: {field_name} (原类型: {dam_type})")
            
            # 过滤特定类型的水坝
            dams_of_type = joined[joined['dam_type'] == dam_type]
            
            # 对每个流域ID计数
            counts = dams_of_type.groupby('HYBAS_ID').size().reset_index(name=field_name)
            
            # 合并到结果DataFrame
            result_df = result_df.merge(counts, on='HYBAS_ID', how='left')
            
            # 填充缺失值为0
            result_df[field_name] = result_df[field_name].fillna(0).astype(int)
        else:
            logging.warning(f"未找到水坝类型 {dam_type} 的输出字段映射，跳过该类型")
    
    return result_df

def main():
    setup_logging()
    
    # 定义文件路径
    basin_path = r"D:/修复数据/研究生/project/data/BasinATLAS_Data_v10_shp/BasinATLAS_v10_shp/BasinATLAS_v10_lev04.shp"
    
    
    # 新生成的o2b转换后的txt文件路径
    dam_txt_paths = {
        2010: r"E:\wyj\dam\v4\global_v7\all_2010_geo_0.4_filtered_all_merged.txt",
        2015: r"E:\wyj\dam\v4\global_v7\all_2015_geo_0.4_filtered_all_merged.txt",
        2020: r"E:\wyj\dam\v4\global_v7\all_2020_geo_0.4_filtered_all_merged.txt"
    }
    
    output_path = r"E:\wyj\project\dam\result1\output\BasinATLAS_v10_lev04_with_dam_stats_v6.shp"
    
    # 水坝类型的输出字段简写映射
    type_abbreviations = {
        'gravity_dam': 'g',
        'Barrage_dam': 'b',  # 包括替换后的 other_dam
        'arch_dam': 'a',
        'embankment_dam': 'e'  # 如果有这种类型
    }
    
    # 水坝类型在文本行中的索引位置
    dam_type_index = 8  # 与o2b文件生成代码中使用的索引一致
    
    # 加载流域shapefile
    logging.info(f"加载流域数据: {basin_path}")
    basins = gpd.read_file(basin_path)
    
    # 确保流域ID的唯一性
    basins = basins.drop_duplicates(subset=['HYBAS_ID'])
    logging.info(f"已加载 {len(basins)} 个流域")
    
    # 结果GeoDataFrame初始化为流域GeoDataFrame的副本
    result = basins.copy()
    
    # 处理每个年份的水坝数据
    for year, dam_path in dam_txt_paths.items():
        if not os.path.exists(dam_path):
            logging.warning(f"水坝数据文件不存在: {dam_path}")
            continue
            
        # 加载水坝数据
        dams = load_dams_from_txt(dam_path, dam_type_index)
        
        if dams.empty:
            logging.warning(f"年份 {year} 没有加载到水坝数据，跳过")
            continue
        
        # 统计流域中的水坝数量
        counts_df = count_dams_in_basins(basins, dams, year, type_abbreviations)
        
        # 合并到结果GeoDataFrame
        for col in counts_df.columns:
            if col != 'HYBAS_ID':  # 跳过ID列
                result[col] = counts_df[col]
                logging.info(f"已添加字段: {col}")
    
    # 保存结果到新的shapefile
    logging.info(f"保存结果到: {output_path}")
    result.to_file(output_path, encoding='utf-8')
    logging.info("处理完成")
    
    # 打印字段统计
    field_summary = {col: [result[col].sum(), result[col].max()] 
                    for col in result.columns 
                    if col not in ['HYBAS_ID', 'geometry'] and col.startswith(('2010', '2015', '2020'))}
    
    logging.info("字段统计信息 (总数, 最大值):")
    for field, [total, maximum] in field_summary.items():
        logging.info(f"{field}: 总数={total}, 最大值={maximum}")

if __name__ == "__main__":
    main()