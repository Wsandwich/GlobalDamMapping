import geopandas as gpd
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import rasterio
from rasterio.plot import show
from matplotlib.colors import Normalize
from rasterio.warp import calculate_default_transform, reproject, Resampling
import logging
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib as mpl

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义全局变量
years = [2010, 2015, 2020]  # 年份列表
dam_types = {
    'gravity_dam': 'gravi',
    'embankment_dam': 'emban',
    'Barrage_dam': 'Barra',
    'arch_dam': 'arch_'
}
resolution = 1.0  # 度

# 定义输入输出路径
input_dir = 'E:/wyj/dam/v4/global_v5_tif/1.0'  # 栅格数据路径，根据您的实际路径修改
output_dir = 'E:/wyj/dam/v4/image/result2/maps'  # 输出地图路径
os.makedirs(output_dir, exist_ok=True)

def load_raster_data(file_path):
    """
    加载栅格数据并返回数据数组和相应的元数据。
    
    参数:
        file_path (str): 栅格文件路径
        
    返回:
        tuple: (data_array, metadata, transform, crs, bounds) 栅格数据、元数据、变换、坐标系和边界
    """
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)  # 读取第一个波段
            meta = src.meta.copy()
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            logging.info(f"栅格 {file_path} 的形状: {data.shape}")
            logging.info(f"栅格边界: {bounds}")
        return data, meta, transform, crs, bounds
    except Exception as e:
        logging.error(f"读取栅格数据时出错: {e}")
        return None, None, None, None, None

def reproject_raster(src_data, src_transform, src_crs, target_shape, target_transform, target_crs):
    """
    将栅格数据重投影到目标形状和投影。
    
    参数:
        src_data (numpy.ndarray): 源数据数组
        src_transform (affine.Affine): 源数据的仿射变换
        src_crs (rasterio.crs.CRS): 源数据的坐标参考系统
        target_shape (tuple): 目标形状 (height, width)
        target_transform (affine.Affine): 目标的仿射变换
        target_crs (rasterio.crs.CRS): 目标的坐标参考系统
        
    返回:
        numpy.ndarray: 重投影后的数据
    """
    dst_data = np.zeros(target_shape, dtype=src_data.dtype)
    
    reproject(
        source=src_data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.nearest)
    
    return dst_data

def ensure_same_dimensions(data_2020, meta_2020, transform_2020, crs_2020, bounds_2020,
                          data_2010, meta_2010, transform_2010, crs_2010, bounds_2010):
    """
    确保两个数据集具有相同的尺寸，如果不同则进行重投影。
    
    返回:
        tuple: (aligned_data_2020, aligned_data_2010, bounds) 对齐后的数据和边界
    """
    if data_2020 is None or data_2010 is None:
        logging.error("数据集为空，无法对齐")
        return None, None, None
    
    # 检查形状是否相同
    if data_2020.shape == data_2010.shape:
        logging.info("数据集形状相同，无需重投影")
        return data_2020, data_2010, bounds_2020
    
    logging.info(f"数据集形状不同: 2020: {data_2020.shape}, 2010: {data_2010.shape}，进行重投影")
    
    # 决定使用哪个形状作为目标
    # 这里选择较大的形状以避免信息丢失
    target_shape = data_2020.shape
    target_transform = transform_2020
    target_crs = crs_2020
    target_bounds = bounds_2020
    
    # 重投影2010年数据到2020年的形状
    aligned_data_2010 = reproject_raster(
        data_2010, transform_2010, crs_2010,
        target_shape, target_transform, target_crs
    )
    
    logging.info(f"重投影后2010年数据形状: {aligned_data_2010.shape}")
    
    return data_2020, aligned_data_2010, target_bounds

def calculate_change_raster(data_2020, data_2010):
    """
    计算两个时间点之间的栅格变化。
    
    参数:
        data_2020 (numpy.ndarray): 2020年的栅格数据
        data_2010 (numpy.ndarray): 2010年的栅格数据
        
    返回:
        numpy.ndarray: 变化值栅格
    """
    if data_2020 is None or data_2010 is None:
        return None
    
    try:
        # 确保数据类型一致
        data_2020_float = data_2020.astype(float)
        data_2010_float = data_2010.astype(float)
        
        # 计算变化
        change = data_2020_float - data_2010_float
        
        return change
    except Exception as e:
        logging.error(f"计算变化栅格时出错: {e}")
        return None

def normalize_change_data(change_data):
    """
    对变化数据进行归一化处理，使用类似于原始代码的tanh方法。
    
    参数:
        change_data (numpy.ndarray): 变化数据数组
    
    返回:
        numpy.ndarray: 归一化后的变化数据
    """
    if change_data is None:
        return None
    
    # 复制一份数据
    normalized_data = change_data.copy()
    
    # 对非零值进行log变换，保留符号
    mask_nonzero = normalized_data != 0
    abs_values = np.abs(normalized_data[mask_nonzero])
    signs = np.sign(normalized_data[mask_nonzero])
    log_values = np.log1p(abs_values)  # log1p(x) = log(1+x)
    normalized_data[mask_nonzero] = signs * log_values
    
    # 最大值归一化到[-1, 1]区间
    max_abs = np.max(np.abs(normalized_data))
    if max_abs > 0:
        normalized_data = np.tanh(normalized_data / max_abs)
    
    return normalized_data

def plot_dam_change_map(normalized_data, bounds, dam_type, ax, title):
    """
    在给定的轴上绘制水坝变化地图。
    
    参数:
        normalized_data (numpy.ndarray): 归一化后的变化数据
        bounds (tuple): 栅格边界 (left, bottom, right, top)
        dam_type (str): 水坝类型
        ax (matplotlib.axes.Axes): 要绘制的轴
        title (str): 地图标题
    """
    # 添加全球海岸线
    ax.coastlines(resolution='50m', linewidth=0.3, color="black")
    
    # 创建掩码，区分无变化和有变化区域
    mask_no_change = normalized_data == 0
    
    # 设置无变化区域颜色
    no_change_color = 'whitesmoke'
    
    # 创建自定义颜色映射以处理无变化区域
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad('whitesmoke', alpha=0.5)  # 设置NaN的颜色
    
    # 获取栅格边界
    left, bottom, right, top = bounds
    
    # 绘制无变化区域
    ax.imshow(np.where(mask_no_change, 1, np.nan), 
              extent=[left, right, bottom, top],
              transform=ccrs.PlateCarree(),
              cmap='gray', alpha=0.3, vmin=0, vmax=1)
    
    # 获取有变化的数据
    change_data = normalized_data.copy()
    change_data[mask_no_change] = np.nan  # 将无变化区域设为NaN
    
    # 绘制变化区域，使用RdBu_r配色
    im = ax.imshow(change_data, 
                  extent=[left, right, bottom, top],
                  transform=ccrs.PlateCarree(),
                  cmap='RdBu_r', vmin=-1, vmax=1)
    
    # 设置地图范围为全球
    ax.set_global()
    
    # 去除图框
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 设置网格线但不显示
    ax.gridlines(draw_labels=False, visible=False)
    
    # 设置标题
    ax.set_title(title, fontsize=12)
    
    return im

def main():
    """
    主函数，处理所有水坝类型并生成变化地图。
    """
    logging.info("开始绘制全球水坝变化地图...")
    
    # 增加dpi设置获得更清晰图像
    mpl.rcParams['figure.dpi'] = 150
    
    # 创建一个图像用于总变化
    fig_total, ax_total = plt.subplots(figsize=(20, 16), subplot_kw={'projection': ccrs.Robinson()})
    total_change_combined = None
    total_shape = None
    total_bounds = None
    
    # 处理每种水坝类型
    for dam_type_full, dam_type_abbr in dam_types.items():
        logging.info(f"处理 {dam_type_full} 类型的水坝变化...")
        
        # 构建文件路径
        file_2010 = os.path.join(input_dir, f"dams_points_2010_{dam_type_full}_{resolution}.tif")
        file_2020 = os.path.join(input_dir, f"dams_points_2020_{dam_type_full}_{resolution}.tif")
        
        # 检查文件是否存在
        if not os.path.exists(file_2010) or not os.path.exists(file_2020):
            logging.warning(f"找不到 {dam_type_full} 的栅格文件，跳过...")
            continue
        
        # 加载栅格数据
        data_2010, meta_2010, transform_2010, crs_2010, bounds_2010 = load_raster_data(file_2010)
        data_2020, meta_2020, transform_2020, crs_2020, bounds_2020 = load_raster_data(file_2020)
        
        if data_2010 is None or data_2020 is None:
            logging.warning(f"无法加载 {dam_type_full} 的数据，跳过...")
            continue
        
        # 确保数据集有相同的维度
        aligned_data_2020, aligned_data_2010, bounds = ensure_same_dimensions(
            data_2020, meta_2020, transform_2020, crs_2020, bounds_2020,
            data_2010, meta_2010, transform_2010, crs_2010, bounds_2010
        )
        
        if aligned_data_2020 is None or aligned_data_2010 is None:
            logging.warning(f"无法对齐 {dam_type_full} 的数据，跳过...")
            continue
        
        # 计算变化
        change_data = calculate_change_raster(aligned_data_2020, aligned_data_2010)
        
        # 如果是第一个水坝类型，初始化总变化数组
        if total_change_combined is None and change_data is not None:
            total_change_combined = np.zeros_like(change_data)
            total_shape = change_data.shape
            total_bounds = bounds
        
        # 确保当前变化数据与总变化形状相同
        if change_data is not None and total_change_combined is not None:
            if change_data.shape != total_shape:
                logging.warning(f"{dam_type_full} 的变化数据形状 {change_data.shape} 与总变化形状 {total_shape} 不匹配，跳过添加...")
            else:
                total_change_combined += change_data
        
        # 归一化变化数据
        normalized_change = normalize_change_data(change_data)
        
        if normalized_change is None:
            logging.warning(f"{dam_type_full} 的归一化变化数据为空，跳过...")
            continue
        
        # 创建单独的图像
        fig, ax = plt.subplots(figsize=(20, 16), subplot_kw={'projection': ccrs.Robinson()})
        
        # 绘制地图
        im = plot_dam_change_map(normalized_change, bounds, dam_type_full, ax, 
                           f"{dam_type_full.capitalize()} Dam Change (2020-2010)")
        
        # 添加颜色条
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap='RdBu_r'),
                           ax=ax, orientation='horizontal', pad=0.05, fraction=0.05)
        cbar.set_label('Normalized Change')
        
        # 保存图像
        output_path = os.path.join(output_dir, f'dam_change_{dam_type_abbr}_2020-2010.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"图像已保存至: {output_path}")
        plt.close(fig)
    
    # 处理总变化
    if total_change_combined is not None and total_bounds is not None:
        # 归一化总变化数据
        normalized_total_change = normalize_change_data(total_change_combined)
        
        # 绘制总变化地图
        plot_dam_change_map(normalized_total_change, total_bounds, "total", ax_total, 
                           "Total Dam Change (2020-2010)")
        
        # 添加颜色条
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap='RdBu_r'),
                           ax=ax_total, orientation='horizontal', pad=0.05, fraction=0.05)
        cbar.set_label('Normalized Change')
        
        # 保存总变化图像
        output_path_total = os.path.join(output_dir, 'dam_change_total_2020-2010.png')
        plt.savefig(output_path_total, dpi=300, bbox_inches='tight')
        logging.info(f"总变化图像已保存至: {output_path_total}")
        plt.close(fig_total)
    
    logging.info("绘图完成!")

if __name__ == "__main__":
    main()