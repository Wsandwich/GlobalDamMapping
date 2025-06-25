import geopandas as gpd
import pandas as pd
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def replace_shapefile_attributes():
    """
    将源shapefile中的指定属性值替换到目标shapefile中
    """
    
    # 文件路径
    source_shp = r"E:\wyj\project\dam\result1\output\BasinATLAS_v10_lev03_with_dam_stats_v6.shp"
    target_shp = r"E:\wyj\project\dam\result2\data_process\output\BasinATLAS_v10_lev03_with_stats_v5.shp"
    output_shp = r"E:\wyj\project\dam\result2\data_process\output\BasinATLAS_v10_lev03_with_stats_v6.shp"
    
    # 需要替换的属性列表
    attributes_to_replace = [
        '2010_e', '2010_g', '2010_b', '2010_a',
        '2015_e', '2015_b', '2015_g', '2015_a', 
        '2020_e', '2020_b', '2020_g', '2020_a'
    ]
    
    try:
        # 检查文件是否存在
        if not Path(source_shp).exists():
            logging.error(f"源文件不存在: {source_shp}")
            return False
            
        if not Path(target_shp).exists():
            logging.error(f"目标文件不存在: {target_shp}")
            return False
        
        # 读取源shapefile和目标shapefile
        logging.info("正在读取源shapefile...")
        source_gdf = gpd.read_file(source_shp)
        logging.info(f"源文件记录数: {len(source_gdf)}")
        
        logging.info("正在读取目标shapefile...")
        target_gdf = gpd.read_file(target_shp)
        logging.info(f"目标文件记录数: {len(target_gdf)}")
        
        # 检查两个文件的记录数是否一致
        if len(source_gdf) != len(target_gdf):
            logging.warning(f"警告: 源文件记录数({len(source_gdf)})与目标文件记录数({len(target_gdf)})不一致")
        
        # 检查属性是否存在
        missing_attrs_source = []
        missing_attrs_target = []
        
        for attr in attributes_to_replace:
            if attr not in source_gdf.columns:
                missing_attrs_source.append(attr)
            if attr not in target_gdf.columns:
                missing_attrs_target.append(attr)
        
        if missing_attrs_source:
            logging.error(f"源文件缺少以下属性: {missing_attrs_source}")
            return False
            
        if missing_attrs_target:
            logging.error(f"目标文件缺少以下属性: {missing_attrs_target}")
            return False
        
        # 显示替换前的统计信息
        logging.info("替换前目标文件的属性统计:")
        for attr in attributes_to_replace:
            if attr in target_gdf.columns:
                total_count = target_gdf[attr].sum()
                max_value = target_gdf[attr].max()
                logging.info(f"{attr}: 总数={total_count}, 最大值={max_value}")
        
        # 执行属性值替换
        logging.info("开始替换属性值...")
        for attr in attributes_to_replace:
            # 将源文件的属性值复制到目标文件
            target_gdf[attr] = source_gdf[attr].values
            logging.info(f"已替换属性: {attr}")
        
        # 显示替换后的统计信息
        logging.info("替换后目标文件的属性统计:")
        for attr in attributes_to_replace:
            total_count = target_gdf[attr].sum()
            max_value = target_gdf[attr].max()
            logging.info(f"{attr}: 总数={total_count}, 最大值={max_value}")
        
        # 创建输出目录
        output_path = Path(output_shp)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存更新后的shapefile
        logging.info(f"正在保存到: {output_shp}")
        target_gdf.to_file(output_shp)
        
        logging.info("属性值替换完成!")
        return True
        
    except Exception as e:
        logging.error(f"处理过程中发生错误: {e}")
        return False

def verify_replacement():
    """
    验证替换是否成功
    """
    
    source_shp = r"E:\wyj\project\dam\result1\output\BasinATLAS_v10_lev03_with_dam_stats_v6.shp"
    output_shp = r"E:\wyj\project\dam\result2\data_process\output\BasinATLAS_v10_lev03_with_stats_v5_updated.shp"
    
    attributes_to_check = [
        '2010_e', '2010_g', '2010_b', '2010_a',
        '2015_e', '2015_b', '2015_g', '2015_a', 
        '2020_e', '2020_b', '2020_g', '2020_a'
    ]
    
    try:
        if not Path(output_shp).exists():
            logging.error(f"输出文件不存在: {output_shp}")
            return False
        
        # 读取源文件和输出文件
        source_gdf = gpd.read_file(source_shp)
        output_gdf = gpd.read_file(output_shp)
        
        logging.info("验证属性值是否替换成功:")
        
        all_match = True
        for attr in attributes_to_check:
            if attr in source_gdf.columns and attr in output_gdf.columns:
                # 比较属性值是否相同
                if source_gdf[attr].equals(output_gdf[attr]):
                    logging.info(f"✓ {attr}: 替换成功")
                else:
                    logging.error(f"✗ {attr}: 替换失败")
                    all_match = False
            else:
                logging.warning(f"? {attr}: 属性不存在于某个文件中")
        
        if all_match:
            logging.info("所有属性值替换验证通过!")
        else:
            logging.error("部分属性值替换验证失败!")
            
        return all_match
        
    except Exception as e:
        logging.error(f"验证过程中发生错误: {e}")
        return False

def main():
    """
    主函数
    """
    logging.info("开始执行shapefile属性值替换...")
    
    # 执行替换
    success = replace_shapefile_attributes()
    
    if success:
        # 验证替换结果
        logging.info("开始验证替换结果...")
        verify_replacement()
    else:
        logging.error("属性值替换失败!")

if __name__ == "__main__":
    main()