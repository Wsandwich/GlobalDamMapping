import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from matplotlib.ticker import ScalarFormatter
import logging

def setup_logging(log_file='hdi_boxplot_analysis.log'):
    """设置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_and_prepare_data(file_path):
    """加载和预处理数据"""
    logging.info(f"加载流域数据: {file_path}")
    gdf = gpd.read_file(file_path)
    logging.info(f"已加载 {len(gdf)} 个流域")
    
    # 确保HDI列存在且为数值型
    if 'hdi_ix_sav' not in gdf.columns:
        logging.error("数据中缺少hdi_ix_sav列")
        return None
    
    gdf['hdi_ix_sav'] = pd.to_numeric(gdf['hdi_ix_sav'], errors='coerce')
    
    # 确保lka_pc_sse列存在且为数值型
    if 'lka_pc_sse' not in gdf.columns:
        logging.error("数据中缺少lka_pc_sse列")
        return None
    
    gdf['lka_pc_sse'] = pd.to_numeric(gdf['lka_pc_sse'], errors='coerce')
    
    # 检查lka_pc_sse为零或负值的情况
    zero_area_count = len(gdf[gdf['lka_pc_sse'] <= 0])
    if zero_area_count > 0:
        logging.warning(f"发现 {zero_area_count} 个流域的lka_pc_sse为零或负值，将被排除在密度计算之外")
        gdf = gdf[gdf['lka_pc_sse'] > 0]
    
    return gdf

def calculate_dam_density(gdf):
    """计算水坝密度"""
    logging.info("计算水坝密度指标")
    
    # 定义水坝类型和年份
    dam_types = ['e', 'g', 'b', 'a']
    years = [2010, 2015, 2020]
    
    # 计算各年份各类型水坝的密度（每平方公里）
    for year in years:
        for dam_type in dam_types:
            column = f'{year}_{dam_type}'
            if column in gdf.columns:
                # 确保值为数值类型
                gdf[column] = pd.to_numeric(gdf[column], errors='coerce').fillna(0).astype(int)
                density_column = f'{year}_{dam_type}_density'
                gdf[density_column] = gdf[column] / gdf['lka_pc_sse']
    
    # 计算2010-2020年密度增量
    for dam_type in dam_types:
        density_2010 = f'2010_{dam_type}_density'
        density_2020 = f'2020_{dam_type}_density'
        increment_col = f'increment_10to20_{dam_type}_density'
        
        if density_2010 in gdf.columns and density_2020 in gdf.columns:
            gdf[increment_col] = gdf[density_2020] - gdf[density_2010]
    
    return gdf

def create_hdi_dam_boxplots(gdf, output_dir):
    """
    为HDI指标创建4个独立的水坝密度箱型图，每种水坝类型一个图
    
    参数:
        gdf (GeoDataFrame): 包含流域和水坝数据的GeoDataFrame
        output_dir (str): 输出图像的目录
    """
    logging.info("为HDI指标创建水坝密度箱型图...")
    
    # 设置图表风格
    plt.style.use('default')
    sns.set_context("paper", font_scale=1.2)
    
    # 按HDI中位数分组
    hdi_median = gdf['hdi_ix_sav'].median()
    gdf['hdi_group'] = np.where(gdf['hdi_ix_sav'] >= hdi_median, 'High HDI', 'Low HDI')
    
    logging.info(f"HDI中位数: {hdi_median:.4f}")
    logging.info(f"高HDI组样本数: {len(gdf[gdf['hdi_group'] == 'High HDI'])}")
    logging.info(f"低HDI组样本数: {len(gdf[gdf['hdi_group'] == 'Low HDI'])}")
    
    # 创建图表目录
    figures_dir = os.path.join(output_dir, "hdi_boxplots")
    os.makedirs(figures_dir, exist_ok=True)
    
    # 水坝类型配置
    dam_configs = [
        {'type': 'e', 'name': 'Embankment Dam', 'color': '#FF6B6B'},
        {'type': 'g', 'name': 'Gravity Dam', 'color': '#4ECDC4'},
        {'type': 'b', 'name': 'Barrage Dam', 'color': '#45B7D1'},
        {'type': 'a', 'name': 'Arch Dam', 'color': '#96CEB4'}
    ]
    
    # 为每种水坝类型创建箱型图
    for dam_config in dam_configs:
        dam_type = dam_config['type']
        dam_name = dam_config['name']
        dam_color = dam_config['color']
        
        # 准备绘图数据
        metrics_to_plot = [
            {
                'column': f'2020_{dam_type}_density',
                'title': f'{dam_name} Density (2020)',
                'ylabel': 'Dam Density (dams/km²)',
                'filename': f'hdi_{dam_type}_density_2020.png'
            },
            {
                'column': f'increment_10to20_{dam_type}_density',
                'title': f'{dam_name} Density Increment (2010-2020)',
                'ylabel': 'Density Increment (dams/km²)',
                'filename': f'hdi_{dam_type}_density_increment.png'
            }
        ]
        
        for metric_config in metrics_to_plot:
            column = metric_config['column']
            
            if column not in gdf.columns:
                logging.warning(f"列 {column} 不存在，跳过")
                continue
            
            # 获取数据
            high_hdi_data = gdf[gdf['hdi_group'] == 'High HDI'][column].dropna()
            low_hdi_data = gdf[gdf['hdi_group'] == 'Low HDI'][column].dropna()
            
            if len(high_hdi_data) == 0 or len(low_hdi_data) == 0:
                logging.warning(f"数据不足，跳过 {column}")
                continue
            
            # 执行t检验
            t_stat, p_value = stats.ttest_ind(high_hdi_data, low_hdi_data, equal_var=False, nan_policy='omit')
            
            # 准备绘图数据
            plot_data = []
            
            # 添加高HDI组数据
            for value in high_hdi_data:
                plot_data.append({'Group': 'High HDI', 'Value': value})
            
            # 添加低HDI组数据
            for value in low_hdi_data:
                plot_data.append({'Group': 'Low HDI', 'Value': value})
            
            plot_df = pd.DataFrame(plot_data)
            
            # 创建箱型图
            plt.figure(figsize=(8, 6))
            
            # 自定义箱型图样式
            boxprops = dict(linewidth=2.0, facecolor='lightblue', alpha=0.7)
            whiskerprops = dict(linewidth=2.0)
            capprops = dict(linewidth=2.0)
            medianprops = dict(linewidth=2.0, color='red')
            flierprops = dict(markersize=4, markerfacecolor='red', alpha=0.5)
            
            # 绘制箱型图
            ax = sns.boxplot(x='Group', y='Value', data=plot_df, 
                           palette=[dam_color, '#CCCCCC'], 
                           width=0.6, showfliers=True,
                           boxprops=boxprops, whiskerprops=whiskerprops, 
                           capprops=capprops, medianprops=medianprops,
                           flierprops=flierprops)
            
            # 调整边框和刻度线
            for spine in ax.spines.values():
                spine.set_linewidth(2.0)
            
            ax.tick_params(axis='both', width=2.0, length=6.0)
            
            # 添加统计信息到图上
            high_mean = high_hdi_data.mean()
            low_mean = low_hdi_data.mean()
            high_median = high_hdi_data.median()
            low_median = low_hdi_data.median()
            
            # 在图上添加统计信息
            stats_text = f'High HDI: Mean={high_mean:.4f}, Median={high_median:.4f}\n'
            stats_text += f'Low HDI: Mean={low_mean:.4f}, Median={low_median:.4f}\n'
            stats_text += f'T-test: t={t_stat:.3f}, p={p_value:.4f}'
            
            plt.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='white', alpha=0.8), fontsize=10)
            
            # 设置标题和标签
            plt.title(metric_config['title'], fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('HDI Group', fontsize=12)
            plt.ylabel(metric_config['ylabel'], fontsize=12)
            
            # Y轴格式化
            if plot_df['Value'].abs().max() < 0.01:
                ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图片
            fig_path = os.path.join(figures_dir, metric_config['filename'])
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"已保存 {dam_name} 图表: {fig_path}")
            logging.info(f"{dam_name} - {metric_config['title']}: t={t_stat:.4f}, p={p_value:.6f}")

def main():
    """主函数"""
    setup_logging()
    
    # 定义文件路径
    basin_file = r"E:\wyj\project\dam\result1\output\BasinATLAS_v10_lev03_with_dam_stats_v6.shp"
    output_dir = r"E:\wyj\project\dam\result1\density_analysis_hdi_lka"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载和预处理数据
    gdf = load_and_prepare_data(basin_file)
    if gdf is None:
        logging.error("数据加载失败，程序终止")
        return
    
    # 计算水坝密度
    gdf = calculate_dam_density(gdf)
    
    # 创建HDI水坝密度箱型图
    create_hdi_dam_boxplots(gdf, output_dir)
    
    logging.info("HDI水坝密度箱型图绘制完成")

if __name__ == "__main__":
    main()