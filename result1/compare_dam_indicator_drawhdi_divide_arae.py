import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from tabulate import tabulate
import logging

def setup_logging(log_file='basin_dam_density_analysis.log'):
    """设置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_basin_data(file_path):
    """加载流域数据"""
    logging.info(f"加载流域数据: {file_path}")
    gdf = gpd.read_file(file_path)
    logging.info(f"已加载 {len(gdf)} 个流域")
    return gdf

def calculate_dam_density_metrics(gdf):
    """计算每个流域的水坝密度和密度增量"""
    # 定义水坝类型和年份
    dam_types = ['e', 'g', 'b', 'a']
    years = [2010, 2015, 2020]
    
    logging.info("计算水坝密度指标")
    
    # 确保lka_pc_sse列存在且为数值型
    if 'lka_pc_sse' not in gdf.columns:
        logging.error("数据中缺少lka_pc_sse列")
        return gdf
    
    gdf['lka_pc_sse'] = pd.to_numeric(gdf['lka_pc_sse'], errors='coerce')
    
    # 检查lka_pc_sse为零或负值的情况
    zero_area_count = len(gdf[gdf['lka_pc_sse'] <= 0])
    if zero_area_count > 0:
        logging.warning(f"发现 {zero_area_count} 个流域的lka_pc_sse为零或负值，将被排除在密度计算之外")
        gdf = gdf[gdf['lka_pc_sse'] > 0]
    
    # 计算各年份水坝总数
    for year in years:
        total_column = f'{year}_total'
        gdf[total_column] = 0
        for dam_type in dam_types:
            column = f'{year}_{dam_type}'
            if column in gdf.columns:
                # 确保值为数值类型
                gdf[column] = pd.to_numeric(gdf[column], errors='coerce').fillna(0).astype(int)
                gdf[total_column] += gdf[column]
    
    # 计算各年份各类型水坝的密度（每平方公里）
    for year in years:
        for dam_type in dam_types:
            column = f'{year}_{dam_type}'
            if column in gdf.columns:
                density_column = f'{year}_{dam_type}_density'
                gdf[density_column] = gdf[column] / gdf['lka_pc_sse']
        
        # 计算总水坝密度
        total_column = f'{year}_total'
        total_density_column = f'{year}_total_density'
        gdf[total_density_column] = gdf[total_column] / gdf['lka_pc_sse']
    
    # 计算2010-2020和2015-2020的密度增量（绝对值差异）
    for dam_type in dam_types:
        # 2010到2020的密度增量
        density_2010 = f'2010_{dam_type}_density'
        density_2020 = f'2020_{dam_type}_density'
        increment_col = f'increment_10to20_{dam_type}_density'
        
        if density_2010 in gdf.columns and density_2020 in gdf.columns:
            gdf[increment_col] = gdf[density_2020] - gdf[density_2010]
        
        # 2015到2020的密度增量
        density_2015 = f'2015_{dam_type}_density'
        increment_col = f'increment_15to20_{dam_type}_density'
        
        if density_2015 in gdf.columns and density_2020 in gdf.columns:
            gdf[increment_col] = gdf[density_2020] - gdf[density_2015]
    
    # 计算总水坝密度增量
    gdf['increment_10to20_total_density'] = gdf['2020_total_density'] - gdf['2010_total_density']
    gdf['increment_15to20_total_density'] = gdf['2020_total_density'] - gdf['2015_total_density']
    
    # 计算年均密度增量率
    gdf['annual_increment_10to20_total_density'] = gdf['increment_10to20_total_density'] / 10
    gdf['annual_increment_15to20_total_density'] = gdf['increment_15to20_total_density'] / 5
    
    # 为各类型水坝计算年均密度增量率
    for dam_type in dam_types:
        increment_10to20 = f'increment_10to20_{dam_type}_density'
        increment_15to20 = f'increment_15to20_{dam_type}_density'
        
        if increment_10to20 in gdf.columns:
            gdf[f'annual_increment_10to20_{dam_type}_density'] = gdf[increment_10to20] / 10
        
        if increment_15to20 in gdf.columns:
            gdf[f'annual_increment_15to20_{dam_type}_density'] = gdf[increment_15to20] / 5
    
    return gdf

def analyze_by_indicator_density(gdf, indicator, output_dir):
    """按指标中位数分组分析流域水坝密度特征"""
    logging.info(f"按指标 {indicator} 分析流域水坝密度")
    
    # 确保指标列是数值型
    gdf[indicator] = pd.to_numeric(gdf[indicator], errors='coerce')
    
    # 计算指标的中位数
    median_value = gdf[indicator].median()
    logging.info(f"指标 {indicator} 的中位数: {median_value}")
    
    # 按指标中位数分组
    gdf['group'] = np.where(gdf[indicator] >= median_value, 'High', 'Low')
    
    # 定义要分析的指标
    dam_types = ['e', 'g', 'b', 'a']
    metrics = []
    
    # 2020年各类型水坝密度
    for dam_type in dam_types:
        metrics.append(f'2020_{dam_type}_density')
    
    # 2010-2020年密度增量
    for dam_type in dam_types:
        metrics.append(f'increment_10to20_{dam_type}_density')
    
    # 2015-2020年密度增量
    for dam_type in dam_types:
        metrics.append(f'increment_15to20_{dam_type}_density')
    
    # 年均密度增量率
    for dam_type in dam_types:
        metrics.append(f'annual_increment_10to20_{dam_type}_density')
        metrics.append(f'annual_increment_15to20_{dam_type}_density')
    
    # 总密度和总增量
    metrics.extend([
        '2020_total_density',
        'increment_10to20_total_density',
        'increment_15to20_total_density',
        'annual_increment_10to20_total_density',
        'annual_increment_15to20_total_density'
    ])
    
    # 按组统计指标并进行显著性检验
    results = []
    
    for metric in metrics:
        if metric in gdf.columns:
            # 按组分割数据
            high_group = gdf[gdf['group'] == 'High'][metric].dropna()
            low_group = gdf[gdf['group'] == 'Low'][metric].dropna()
            
            # 计算各组的均值和中位数
            high_mean = high_group.mean()
            low_mean = low_group.mean()
            high_median = high_group.median()
            low_median = low_group.median()
            
            # 执行t检验
            if len(high_group) > 0 and len(low_group) > 0:
                t_stat, p_value = stats.ttest_ind(high_group, low_group, equal_var=False, nan_policy='omit')
                significant = "是" if p_value < 0.05 else "否"
            else:
                t_stat, p_value = np.nan, np.nan
                significant = "数据不足"
            
            # 记录结果
            results.append({
                'Metric': metric,
                'High_Mean': high_mean,
                'Low_Mean': low_mean,
                'High_Median': high_median,
                'Low_Median': low_median,
                'Difference_Mean': high_mean - low_mean,
                'T_Statistic': t_stat,
                'P_Value': p_value,
                'Significant': significant
            })
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 格式化输出
    formatted_results = []
    for i, row in results_df.iterrows():
        formatted_row = [
            row['Metric'],
            f"{row['High_Mean']:.6f}",
            f"{row['Low_Mean']:.6f}",
            f"{row['High_Median']:.6f}",
            f"{row['Low_Median']:.6f}",
            f"{row['Difference_Mean']:.6f}",
            f"{row['T_Statistic']:.3f}" if not pd.isna(row['T_Statistic']) else "N/A",
            f"{row['P_Value']:.6f}" if not pd.isna(row['P_Value']) else "N/A",
            row['Significant']
        ]
        formatted_results.append(formatted_row)
    
    # 创建表格
    headers = ["指标", "高组均值", "低组均值", "高组中位数", "低组中位数", "均值差异", "T统计量", "P值", "显著差异"]
    table = tabulate(formatted_results, headers, tablefmt="grid")
    
    # 保存结果
    result_file = os.path.join(output_dir, f"density_analysis_{indicator}.txt")
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"按指标 {indicator} 分组的水坝密度分析结果\n")
        f.write(f"指标中位数: {median_value}\n")
        f.write(f"高组样本数: {len(gdf[gdf['group'] == 'High'])}\n")
        f.write(f"低组样本数: {len(gdf[gdf['group'] == 'Low'])}\n\n")
        f.write("说明:\n")
        f.write("- 密度单位: 水坝数量/km²\n")
        f.write("- 增量单位: 水坝数量/km²的变化量\n")
        f.write("- 年均增量单位: 水坝数量/km²/年\n")
        f.write("- 正值表示密度增加，负值表示密度减少\n\n")
        f.write(table)
    
    logging.info(f"已将 {indicator} 密度分析结果保存至 {result_file}")
    
    # 还可以保存为CSV格式，便于后续处理
    csv_file = os.path.join(output_dir, f"density_analysis_{indicator}.csv")
    results_df.to_csv(csv_file, index=False)
    
    return results_df

def create_density_boxplots(gdf, indicator, output_dir):
    """
    为指定指标创建水坝密度特征的箱型图
    
    参数:
        gdf (GeoDataFrame): 包含流域和水坝数据的GeoDataFrame
        indicator (str): 分组指标名称
        output_dir (str): 输出图像的目录
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import ScalarFormatter
    import numpy as np
    import os
    
    logging.info(f"为指标 {indicator} 创建水坝密度箱型图...")
    
    # 设置图表风格
    plt.style.use('default')
    sns.set_context("paper", font_scale=1.2)
    
    # 确保指标字段是数值型
    gdf[indicator] = pd.to_numeric(gdf[indicator], errors='coerce')
    
    # 按指标中位数分组
    median_value = gdf[indicator].median()
    gdf['group'] = np.where(gdf[indicator] >= median_value, f'High {indicator}', f'Low {indicator}')
    
    # 移除无效值
    gdf = gdf.dropna(subset=['group'])
    
    # 创建图表目录
    figures_dir = os.path.join(output_dir, "density_figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # 定义要绘制的指标
    plot_configs = [
        {
            'metrics': ['2020_g_density', '2020_b_density', '2020_e_density'],
            'title': f'Dam Density by {indicator} (2020)',
            'ylabel': 'Dam Density (dams/km²)',
            'filename': f'{indicator}_dam_density_2020.png',
            'labels': ['Gravity', 'Barrage', 'Embankment']
        },
        {
            'metrics': ['increment_10to20_g_density', 'increment_10to20_b_density'],
            'title': f'Dam Density Increment by {indicator} (2010-2020)',
            'ylabel': 'Density Increment (dams/km²)',
            'filename': f'{indicator}_density_increment_10to20.png',
            'labels': ['Gravity', 'Barrage']
        },
        {
            'metrics': ['annual_increment_10to20_g_density', 'annual_increment_10to20_b_density'],
            'title': f'Annual Dam Density Increment by {indicator} (2010-2020)',
            'ylabel': 'Annual Density Increment (dams/km²/year)',
            'filename': f'{indicator}_annual_density_increment.png',
            'labels': ['Gravity', 'Barrage']
        }
    ]
    
    for plot_config in plot_configs:
        # 准备数据
        plot_data_list = []
        
        for i, metric in enumerate(plot_config['metrics']):
            if metric not in gdf.columns:
                continue
                
            # 从gdf中提取每个组的数据
            high_group = gdf[gdf['group'] == f'High {indicator}'][metric].dropna()
            low_group = gdf[gdf['group'] == f'Low {indicator}'][metric].dropna()
            
            # 添加到数据列表
            for value in high_group:
                plot_data_list.append({
                    'Group': f'High {indicator}',
                    'Value': value,
                    'Metric': plot_config['labels'][i]
                })
            
            for value in low_group:
                plot_data_list.append({
                    'Group': f'Low {indicator}',
                    'Value': value,
                    'Metric': plot_config['labels'][i]
                })
            
            # 执行t检验并记录结果
            if len(high_group) > 0 and len(low_group) > 0:
                t_stat, p_value = stats.ttest_ind(high_group, low_group, equal_var=False, nan_policy='omit')
                logging.info(f"T-test for {metric}: t={t_stat:.4f}, p={p_value:.6f}")
        
        if not plot_data_list:
            logging.warning(f"没有数据可用于绘制 {plot_config['filename']}")
            continue
        
        # 创建DataFrame
        plot_df = pd.DataFrame(plot_data_list)
        
        # 创建箱型图
        plt.figure(figsize=(10, 6))
        
        # 设置箱型图样式
        boxprops = dict(linewidth=2.0)
        whiskerprops = dict(linewidth=2.0)
        capprops = dict(linewidth=2.0)
        medianprops = dict(linewidth=2.0)
        flierprops = dict(markersize=3)
        
        # 使用seaborn绘制箱型图
        ax = sns.boxplot(x='Metric', y='Value', hue='Group', data=plot_df, 
                        palette="Set2", width=0.7, showfliers=True,
                        boxprops=boxprops, whiskerprops=whiskerprops, 
                        capprops=capprops, medianprops=medianprops,
                        flierprops=flierprops)
        
        # 调整边框和刻度线
        ax.spines['bottom'].set_linewidth(2.0)
        ax.spines['left'].set_linewidth(2.0)
        ax.spines['top'].set_linewidth(2.0)
        ax.spines['right'].set_linewidth(2.0)
        
        ax.tick_params(axis='x', width=2.0, length=10.0)
        ax.tick_params(axis='y', width=2.0, length=10.0)
        
        # 添加标题和标签
        plt.title(plot_config['title'], fontsize=14, fontweight='bold')
        plt.xlabel('')
        plt.ylabel(plot_config['ylabel'], fontsize=12)
        
        # Y轴格式化
        if plot_df['Value'].abs().max() < 0.01:
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        fig_path = os.path.join(figures_dir, plot_config['filename'])
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"已保存图表: {fig_path}")

def generate_density_summary_report(all_results, output_dir):
    """生成水坝密度分析的汇总报告"""
    logging.info("生成水坝密度分析汇总报告")
    
    # 指标说明字典
    indicator_descriptions = {
        'dis_m3_pyr': "年平均径流量(立方米/年)",
        'run_mm_syr': "年径流深度(毫米/年)",
        'gwt_cm_sav': "地下水位深度(厘米)",
        'pre_mm_syr': "年降水量(毫米/年)",
        'hdi_ix_sav': "人类发展指数",
        'ari_ix_sav': "干旱指数"
    }
    
    # 创建汇总报告文件
    summary_file = os.path.join(output_dir, "density_summary_report.txt")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("流域水坝密度分析汇总报告\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("报告概述\n")
        f.write("-" * 50 + "\n")
        f.write("本报告分析了不同类型流域中水坝密度特征的差异。与传统的百分比变化率不同，\n")
        f.write("本分析重点关注水坝密度的绝对增量（每平方公里水坝数量的实际变化）。\n")
        f.write("这种方法更能反映水坝建设对流域景观的实际影响强度。\n\n")
        
        f.write("分析方法说明\n")
        f.write("-" * 50 + "\n")
        f.write("1. 水坝密度 = 水坝数量 / 流域面积(km²)\n")
        f.write("2. 密度增量 = 后期密度 - 前期密度（单位：水坝数量/km²）\n")
        f.write("3. 年均密度增量 = 密度增量 / 年数（单位：水坝数量/km²/年）\n")
        f.write("4. 流域按各指标中位数分为高值组和低值组\n")
        f.write("5. 显著性水平设定为p<0.05\n\n")
        
        # 分析结果摘要
        f.write("分析结果摘要\n")
        f.write("=" * 100 + "\n\n")
        
        significant_found = False
        
        # 对每个指标进行分析
        for indicator, results in all_results.items():
            f.write(f"\n指标: {indicator} ({indicator_descriptions.get(indicator, '无描述')})\n")
            f.write("-" * 100 + "\n")
            
            # 筛选出显著差异的指标
            significant_results = results[results['P_Value'] < 0.05]
            
            if len(significant_results) > 0:
                significant_found = True
                f.write(f"发现{len(significant_results)}个水坝密度指标存在显著差异:\n\n")
                
                # 分类显示结果
                densities_2020 = []
                increments_10to20 = []
                increments_15to20 = []
                annual_increments = []
                
                for i, row in significant_results.iterrows():
                    metric = row['Metric']
                    high_mean = row['High_Mean']
                    low_mean = row['Low_Mean']
                    diff = row['Difference_Mean']
                    p_value = row['P_Value']
                    
                    result_str = f"{metric}: 高组={high_mean:.6f}, 低组={low_mean:.6f}, 差值={diff:.6f}, p={p_value:.6f}"
                    
                    if metric.endswith('_density') and '2020' in metric:
                        densities_2020.append(result_str)
                    elif metric.startswith('increment_10to20'):
                        increments_10to20.append(result_str)
                    elif metric.startswith('increment_15to20'):
                        increments_15to20.append(result_str)
                    elif metric.startswith('annual_increment'):
                        annual_increments.append(result_str)
                
                # 按类别输出结果
                if densities_2020:
                    f.write("【2020年水坝密度】\n")
                    for result in densities_2020:
                        f.write(f"  - {result}\n")
                    f.write("\n")
                
                if increments_10to20:
                    f.write("【2010-2020年密度增量】\n")
                    for result in increments_10to20:
                        f.write(f"  - {result}\n")
                    f.write("\n")
                
                if increments_15to20:
                    f.write("【2015-2020年密度增量】\n")
                    for result in increments_15to20:
                        f.write(f"  - {result}\n")
                    f.write("\n")
                
                if annual_increments:
                    f.write("【年均密度增量】\n")
                    for result in annual_increments:
                        f.write(f"  - {result}\n")
                    f.write("\n")
                
            else:
                f.write(f"在按{indicator}分组的分析中，未发现水坝密度特征有显著差异。\n\n")
        
        # 总结
        f.write("\n总体结论\n")
        f.write("=" * 100 + "\n")
        if significant_found:
            f.write("水坝密度分析揭示了以下重要发现：\n\n")
            f.write("1. 不同流域特征确实影响水坝建设的密度和增长模式\n")
            f.write("2. 密度增量分析比百分比变化更能反映水坝建设的实际强度\n")
            f.write("3. 某些类型的水坝在特定流域条件下表现出更高的建设密度\n")
            f.write("4. 年均密度增量有助于理解水坝建设的时间趋势\n\n")
        else:
            f.write("基于所有分析指标，水坝密度特征在不同流域类型间无显著差异。\n")
        
        f.write(f"\n分析完成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    logging.info(f"水坝密度分析汇总报告已保存至 {summary_file}")

def main():
    setup_logging()
    
    # 定义文件路径
    basin_file = r"E:\wyj\project\dam\result1\output\BasinATLAS_v10_lev03_with_dam_stats_v6.shp"
    output_dir = r"E:\wyj\project\dam\result1\density_analysis_results_lka"
    os.makedirs(output_dir, exist_ok=True)
    
    # 要分析的指标列表
    indicators = ['dis_m3_pyr', 'run_mm_syr', 'gwt_cm_sav', 'pre_mm_syr', 'hdi_ix_sav', 'ari_ix_sav']
    
    # 加载数据
    gdf = load_basin_data(basin_file)
    
    # 计算水坝密度指标
    gdf = calculate_dam_density_metrics(gdf)
    
    # 保存中间数据以便检查
    interim_file = os.path.join(output_dir, "basin_with_density_metrics.csv")
    gdf.drop('geometry', axis=1).to_csv(interim_file, index=False)
    logging.info(f"密度计算结果已保存至 {interim_file}")
    
    # 存储所有分析结果
    all_results = {}
    
    # 按每个指标分析密度特征
    for indicator in indicators:
        if indicator in gdf.columns:
            results = analyze_by_indicator_density(gdf, indicator, output_dir)
            all_results[indicator] = results
            
            # 为每个指标创建箱型图
            create_density_boxplots(gdf, indicator, output_dir)
        else:
            logging.warning(f"指标 {indicator} 不在数据集中，跳过分析")
    
    # 生成汇总报告
    generate_density_summary_report(all_results, output_dir)
    
    logging.info("水坝密度分析完成")

if __name__ == "__main__":
    main()