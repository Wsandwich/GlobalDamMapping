import os
import glob
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义基础目录
base_dir = "/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/dof_result"
output_path = os.path.join(base_dir, "output", "analysis")
os.makedirs(output_path, exist_ok=True)

# 获取所有 GeoPackage 文件
gpkg_files = glob.glob(os.path.join(base_dir, "*.gpkg"))
logger.info(f"找到 {len(gpkg_files)} 个 GeoPackage 文件")

# 排除 shapefiles（仅处理 .gpkg 文件）
if not gpkg_files:
    logger.error("目录中未找到 GeoPackage 文件")
    exit()

# 函数：读取单个 GeoPackage 文件的 DIS_AV_CMS
def read_dis_av_cms(gpkg_file):
    try:
        # 读取 DOF 图层
        gdf = gpd.read_file(gpkg_file, layer='DOF', usecols=['DIS_AV_CMS'])
        # 确保 DIS_AV_CMS 列存在
        if 'DIS_AV_CMS' not in gdf.columns:
            logger.warning(f"{gpkg_file} 中未找到 DIS_AV_CMS 列")
            return None
        # 提取 DIS_AV_CMS（转换为数值，忽略无效值）
        dis_av_cms = pd.to_numeric(gdf['DIS_AV_CMS'], errors='coerce').dropna().values
        logger.info(f"从 {gpkg_file} 提取 {len(dis_av_cms)} 个 DIS_AV_CMS 值")
        return dis_av_cms
    except Exception as e:
        logger.error(f"读取 {gpkg_file} 失败: {str(e)}")
        return None

# 多线程加载所有 GeoPackage 文件
def load_gpkg_multithreaded(gpkg_files, max_workers=35):
    all_dis_av_cms = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 并行读取所有文件
        results = executor.map(read_dis_av_cms, gpkg_files)
        # 收集结果
        for result in results:
            if result is not None and len(result) > 0:
                all_dis_av_cms.extend(result)
    return np.array(all_dis_av_cms)

# 分析并可视化 DIS_AV_CMS 分布
def analyze_dis_av_cms(dis_av_cms, output_path):
    if len(dis_av_cms) == 0:
        logger.error("未提取到有效的 DIS_AV_CMS 数据")
        return

    # 计算统计信息
    stats = {
        'count': len(dis_av_cms),
        'mean': np.mean(dis_av_cms),
        'median': np.median(dis_av_cms),
        'std': np.std(dis_av_cms),
        'min': np.min(dis_av_cms),
        'max': np.max(dis_av_cms)
    }
    logger.info("DIS_AV_CMS 统计信息：")
    for key, value in stats.items():
        logger.info(f"{key}: {value:.2f}")

    # 保存统计信息到文件
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_file = os.path.join(output_path, f"dis_av_cms_stats_{current_time}.txt")
    with open(stats_file, 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value:.2f}\n")
    logger.info(f"统计信息保存至 {stats_file}")

    # 可视化分布（直方图 + 核密度估计）
    plt.figure(figsize=(10, 6))
    sns.histplot(dis_av_cms, bins=50, kde=True, color='blue')
    plt.title('DIS_AV_CMS Distribution', fontsize=16)
    plt.xlabel('DIS_AV_CMS (Cubic Meters per Second)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 保存分布图
    dist_plot_file = os.path.join(output_path, f"dis_av_cms_distribution_{current_time}.png")
    plt.savefig(dist_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"分布图保存至 {dist_plot_file}")

# 主程序
def main():
    # 多线程加载 DIS_AV_CMS 数据
    dis_av_cms = load_gpkg_multithreaded(gpkg_files, max_workers=4)
    # 分析并可视化分布
    analyze_dis_av_cms(dis_av_cms, output_path)
    logger.info("分析完成！")

if __name__ == "__main__":
    main()