import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from tabulate import tabulate
import logging

def setup_logging(log_file='basin_dam_analysis.log'):
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

def calculate_dam_metrics(gdf):
    """计算每个流域的水坝总数和各类型百分比"""
    # 定义水坝类型和年份
    dam_types = ['e', 'g', 'b', 'a']
    years = [2010, 2015, 2020]
    
    logging.info("计算水坝指标")
    
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
    
    # 计算各年份各类型水坝的百分比
    for year in years:
        for dam_type in dam_types:
            column = f'{year}_{dam_type}'
            if column in gdf.columns:
                percent_column = f'{year}_{dam_type}_pct'
                total_column = f'{year}_total'
                # 避免除以零错误
                gdf[percent_column] = np.where(gdf[total_column] > 0, 
                                             (gdf[column] / gdf[total_column]) * 100, 
                                             0)
    
    # 计算2010-2020和2015-2020的变化率
    for dam_type in dam_types:
        # 2010到2020的变化率
        col_2010 = f'2010_{dam_type}'
        col_2020 = f'2020_{dam_type}'
        change_col = f'change_10to20_{dam_type}'
        
        if col_2010 in gdf.columns and col_2020 in gdf.columns:
            gdf[change_col] = np.where(gdf[col_2010] > 0,
                                     ((gdf[col_2020] - gdf[col_2010]) / gdf[col_2010]) * 100,
                                     np.where(gdf[col_2020] > 0, np.inf, 0))  # 如果2010年为0，2020年大于0，则变化率为无穷大
        
        # 2015到2020的变化率
        col_2015 = f'2015_{dam_type}'
        change_col = f'change_15to20_{dam_type}'
        
        if col_2015 in gdf.columns and col_2020 in gdf.columns:
            gdf[change_col] = np.where(gdf[col_2015] > 0,
                                     ((gdf[col_2020] - gdf[col_2015]) / gdf[col_2015]) * 100,
                                     np.where(gdf[col_2020] > 0, np.inf, 0))
    
    # 同样计算总水坝数量的变化率
    gdf['change_10to20_total'] = np.where(gdf['2010_total'] > 0,
                                        ((gdf['2020_total'] - gdf['2010_total']) / gdf['2010_total']) * 100,
                                        np.where(gdf['2020_total'] > 0, np.inf, 0))
    
    gdf['change_15to20_total'] = np.where(gdf['2015_total'] > 0,
                                        ((gdf['2020_total'] - gdf['2015_total']) / gdf['2015_total']) * 100,
                                        np.where(gdf['2020_total'] > 0, np.inf, 0))
    
    return gdf

def analyze_by_indicator(gdf, indicator, output_dir):
    """按指标中位数分组分析流域"""
    logging.info(f"按指标 {indicator} 分析流域")
    
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
    
    # 2020年各类型水坝数量
    for dam_type in dam_types:
        metrics.append(f'2020_{dam_type}')
    
    # 2020年各类型水坝百分比
    for dam_type in dam_types:
        metrics.append(f'2020_{dam_type}_pct')
    
    # 2010到2020的变化率
    for dam_type in dam_types:
        metrics.append(f'change_10to20_{dam_type}')
    
    # 2015到2020的变化率
    for dam_type in dam_types:
        metrics.append(f'change_15to20_{dam_type}')
    
    # 总数和总变化率
    metrics.extend(['2020_total', 'change_10to20_total', 'change_15to20_total'])
    
    # 按组统计指标并进行显著性检验
    results = []
    
    for metric in metrics:
        if metric in gdf.columns:
            # 替换无穷大值为NaN以便统计
            gdf[metric] = gdf[metric].replace([np.inf, -np.inf], np.nan)
            
            # 按组分割数据
            high_group = gdf[gdf['group'] == 'High'][metric].dropna()
            low_group = gdf[gdf['group'] == 'Low'][metric].dropna()
            
            # 计算各组的均值
            high_mean = high_group.mean()
            low_mean = low_group.mean()
            
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
            f"{row['High_Mean']:.2f}",
            f"{row['Low_Mean']:.2f}",
            f"{row['T_Statistic']:.2f}" if not pd.isna(row['T_Statistic']) else "N/A",
            f"{row['P_Value']:.4f}" if not pd.isna(row['P_Value']) else "N/A",
            row['Significant']
        ]
        formatted_results.append(formatted_row)
    
    # 创建表格
    headers = ["指标", "高组均值", "低组均值", "T统计量", "P值", "显著差异"]
    table = tabulate(formatted_results, headers, tablefmt="grid")
    
    # 保存结果
    result_file = os.path.join(output_dir, f"analysis_{indicator}.txt")
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"按指标 {indicator} 分组分析结果\n")
        f.write(f"指标中位数: {median_value}\n")
        f.write(f"高组样本数: {len(gdf[gdf['group'] == 'High'])}\n")
        f.write(f"低组样本数: {len(gdf[gdf['group'] == 'Low'])}\n\n")
        f.write(table)
    
    logging.info(f"已将 {indicator} 分析结果保存至 {result_file}")
    
    # 还可以保存为CSV格式，便于后续处理
    csv_file = os.path.join(output_dir, f"analysis_{indicator}.csv")
    results_df.to_csv(csv_file, index=False)
    
    return results_df


def create_boxplots_for_hdi(gdf, output_dir):
    """
    为HDI指标创建水坝特征的箱型图
    
    参数:
        gdf (GeoDataFrame): 包含流域和水坝数据的GeoDataFrame
        output_dir (str): 输出图像的目录
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import ScalarFormatter
    import numpy as np
    import os
    
    logging.info("创建HDI指标相关的箱型图...")
    
    # 设置图表风格 - Nature样式
    plt.style.use('default')
    sns.set_context("paper", font_scale=1.2)
    
    # 确保HDI字段是数值型
    gdf['hdi_ix_sav'] = pd.to_numeric(gdf['hdi_ix_sav'], errors='coerce')
    
    # 按HDI中位数分组
    median_hdi = gdf['hdi_ix_sav'].median()
    gdf['hdi_group'] = np.where(gdf['hdi_ix_sav'] >= median_hdi, 'High HDI', 'Low HDI')
    
    # 移除无效值
    gdf = gdf.dropna(subset=['hdi_group'])
    
    # 创建图表目录
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # 定义要绘制的指标
    plot_data = [
        {
            'metrics': ['2020_e_pct', '2020_g_pct', '2020_a_pct'],
            'title': 'Dam Type Distribution by HDI (2020)',
            'ylabel': 'Percentage (%)',
            'filename': 'hdi_dam_type_distribution.png',
            'labels': ['Embankment', 'Gravity', 'Arch']
        },
        {
            'metrics': ['change_10to20_g', 'change_10to20_b'],
            'title': 'Dam Growth Rate by HDI (2010-2020)',
            'ylabel': 'Growth Rate (%)',
            'filename': 'hdi_dam_growth_rate.png',
            'labels': ['Gravity', 'Barrage']
        },
        {
            'metrics': ['change_15to20_g', 'change_15to20_b'],
            'title': 'Recent Dam Growth Rate by HDI (2015-2020)',
            'ylabel': 'Growth Rate (%)',
            'filename': 'hdi_recent_growth_rate.png',
            'labels': ['Gravity', 'Barrage']
        }
    ]

    # 特别处理增长率图表
    growth_plot_info = [p for p in plot_data if p['filename'] == 'hdi_dam_growth_rate.png'][0]
    metrics = growth_plot_info['metrics']
    labels = growth_plot_info['labels']

    # 准备数据
    plot_data_list = []
    outlier_basins = {}  # 存储异常值流域信息

    for i, metric in enumerate(metrics):
        # 从gdf中提取每个组的数据
        high_hdi_df = gdf[gdf['hdi_group'] == 'High HDI']
        low_hdi_df = gdf[gdf['hdi_group'] == 'Low HDI']
        
        high_hdi = high_hdi_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
        low_hdi = low_hdi_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
        
        t_stat, p_value = stats.ttest_ind(high_hdi, low_hdi, equal_var=False, nan_policy='omit')
        sig_text = f"p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
        
        # 添加这里 - 打印每个指标的p值
        print(f"指标 {metric} 的 t 检验结果：t = {t_stat:.4f}, p = {p_value:.8f}")


        # 查找异常值流域
        if metric == 'change_10to20_b':  # 堰坝增长率
            # 查找高HDI组中堰坝增长率最大的流域
            if not high_hdi.empty:
                max_idx = high_hdi.idxmax()
                max_basin_id = high_hdi_df.loc[max_idx, 'HYBAS_ID']
                max_value = high_hdi.loc[max_idx]
                outlier_basins['high_max_barrage'] = {
                    'HYBAS_ID': max_basin_id,
                    'Value': max_value
                }
                logging.info(f"高HDI组堰坝增长率最大的流域: ID={max_basin_id}, 增长率={max_value:.2f}%")
            
            # 查找低HDI组中堰坝增长率最小的流域
            if not low_hdi.empty:
                min_idx = low_hdi.idxmin()
                min_basin_id = low_hdi_df.loc[min_idx, 'HYBAS_ID']
                min_value = low_hdi.loc[min_idx]
                outlier_basins['low_min_barrage'] = {
                    'HYBAS_ID': min_basin_id,
                    'Value': min_value
                }
                logging.info(f"低HDI组堰坝增长率最小的流域: ID={min_basin_id}, 增长率={min_value:.2f}%")
        
        # 添加到数据列表
        for value in high_hdi:
            plot_data_list.append({
                'HDI Group': 'High HDI',
                'Value': value,
                'Metric': labels[i]
            })
        
        for value in low_hdi:
            plot_data_list.append({
                'HDI Group': 'Low HDI',
                'Value': value,
                'Metric': labels[i]
            })
        
        # 执行t检验
        t_stat, p_value = stats.ttest_ind(high_hdi, low_hdi, equal_var=False, nan_policy='omit')
        sig_text = f"p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
        logging.info(f"T-test for {metric}: {sig_text}")

    # 创建DataFrame
    plot_df = pd.DataFrame(plot_data_list)

    # 创建箱型图
    plt.figure(figsize=(10, 6))

    boxprops = dict(linewidth=2.0)       # 箱体边框粗细
    whiskerprops = dict(linewidth=2.0)   # 晶须粗细
    capprops = dict(linewidth=2.0)       # 晶须末端横线粗细
    medianprops = dict(linewidth=2.0)    # 中位数线粗细
    flierprops = dict(markersize=2)      # 异常值点大小（设为0表示不显示）

    # 使用seaborn绘制箱型图，应用自定义样式
    ax = sns.boxplot(x='Metric', y='Value', hue='HDI Group', data=plot_df, 
                    palette="Set2", width=0.7,
                    whis=[0, 100],  # 这是关键修改！让晶须显示0%到100%分位数（即最小值到最大值）
                    showfliers=False,  # 隐藏异常值
                    boxprops=boxprops, whiskerprops=whiskerprops, 
                    capprops=capprops, medianprops=medianprops,
                    flierprops=flierprops)


    # 设置y轴上限为100
    ax.set_ylim(top=200)

    # 调整边框和刻度线的粗细和长度
    ax.spines['bottom'].set_linewidth(2.0)  # x轴
    ax.spines['left'].set_linewidth(2.0)    # y轴
    ax.spines['top'].set_linewidth(2.0)     # 顶部边框
    ax.spines['right'].set_linewidth(2.0)   # 右侧边框

    # 调整刻度线粗细和长度
    ax.tick_params(axis='x', width=2.0, length=10.0)
    ax.tick_params(axis='y', width=2.0, length=10.0)

    # 添加标题和标签
    # plt.title(growth_plot_info['title'], fontsize=14, fontweight='bold')
    plt.xlabel('')
    plt.ylabel(growth_plot_info['ylabel'], fontsize=0)

    # 修改图例
    # plt.legend(title='', loc='best', frameon=True)

    # Y轴格式化
    if plot_df['Value'].max() > 1000:
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # 添加样本量信息
    high_count = len(gdf[gdf['hdi_group'] == 'High HDI'])
    low_count = len(gdf[gdf['hdi_group'] == 'Low HDI'])
    # plt.figtext(0.01, 0.01, f"Sample sizes: High HDI (n={high_count}), Low HDI (n={low_count})", 
    #             ha='left', fontsize=8, style='italic')

    # 添加异常值流域注释
    if outlier_basins:
        outlier_text = "Notable basins:\n"
        if 'high_max_barrage' in outlier_basins:
            basin = outlier_basins['high_max_barrage']
            outlier_text += f"- High HDI max Barrage growth: Basin ID {basin['HYBAS_ID']} ({basin['Value']:.1f}%)\n"
        if 'low_min_barrage' in outlier_basins:
            basin = outlier_basins['low_min_barrage']
            outlier_text += f"- Low HDI min Barrage growth: Basin ID {basin['HYBAS_ID']} ({basin['Value']:.1f}%)"
        
        # plt.figtext(0.5, 0.01, outlier_text, ha='center', fontsize=8, style='italic')

    # 添加说明
    # plt.figtext(0.99, 0.01, 
    #             "Statistical analysis: two-sided Welch's t-test\n" +
    #             "Note: Outliers are not shown, only whiskers represent min/max values.", 
    #             ha='right', fontsize=8, style='italic')

    # 调整布局
    plt.tight_layout()

    # 保存图片
    fig_path = os.path.join(figures_dir, growth_plot_info['filename'])
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"已保存增长率图表（不含异常值点）: {fig_path}")

    # 保存异常值流域信息到文本文件
    if outlier_basins:
        outlier_file = os.path.join(figures_dir, 'outlier_basins.txt')
        with open(outlier_file, 'w', encoding='utf-8') as f:
            f.write("异常值流域信息\n")
            f.write("=" * 50 + "\n\n")
            
            if 'high_max_barrage' in outlier_basins:
                basin = outlier_basins['high_max_barrage']
                f.write(f"高HDI组堰坝增长率最大的流域:\n")
                f.write(f"  流域ID: {basin['HYBAS_ID']}\n")
                f.write(f"  增长率: {basin['Value']:.2f}%\n\n")
                
                # 尝试获取更多流域信息
                basin_info = gdf[gdf['HYBAS_ID'] == basin['HYBAS_ID']]
                if not basin_info.empty:
                    f.write("流域详细信息:\n")
                    for col in ['dis_m3_pyr', 'run_mm_syr', 'gwt_cm_sav', 'pre_mm_syr', 'hdi_ix_sav', 'ari_ix_sav']:
                        if col in basin_info.columns:
                            f.write(f"  {col}: {basin_info[col].iloc[0]}\n")
                    f.write("\n")
            
            if 'low_min_barrage' in outlier_basins:
                basin = outlier_basins['low_min_barrage']
                f.write(f"低HDI组堰坝增长率最小的流域:\n")
                f.write(f"  流域ID: {basin['HYBAS_ID']}\n")
                f.write(f"  增长率: {basin['Value']:.2f}%\n\n")
                
                # 尝试获取更多流域信息
                basin_info = gdf[gdf['HYBAS_ID'] == basin['HYBAS_ID']]
                if not basin_info.empty:
                    f.write("流域详细信息:\n")
                    for col in ['dis_m3_pyr', 'run_mm_syr', 'gwt_cm_sav', 'pre_mm_syr', 'hdi_ix_sav', 'ari_ix_sav']:
                        if col in basin_info.columns:
                            f.write(f"  {col}: {basin_info[col].iloc[0]}\n")
        
        logging.info(f"异常值流域信息已保存至: {outlier_file}")
    
def generate_summary_report(all_results, output_dir):
    """生成所有指标的详细汇总报告，包含指标说明"""
    logging.info("生成详细汇总报告")
    
    # 指标说明字典
    indicator_descriptions = {
        'dis_m3_pyr': "年平均径流量(立方米/年)：表示流域内河流的年平均水流量，是衡量流域水资源丰富程度的重要指标。",
        'run_mm_syr': "年径流深度(毫米/年)：表示流域内降水形成的地表径流厚度，反映了降水转化为径流的能力。",
        'gwt_cm_sav': "地下水位深度(厘米)：表示地下水面到地表的平均深度，影响水资源的可获取性和土壤湿度。",
        'pre_mm_syr': "年降水量(毫米/年)：表示流域内年平均降水总量，是流域水循环的重要输入。",
        'hdi_ix_sav': "人类发展指数：综合反映一个地区在健康、教育和收入方面的发展水平，范围为0-1000，数值越高表示发展水平越高。",
        'ari_ix_sav': "干旱指数：反映地区的干旱程度，数值越低表示地区越干旱。"
    }
    
    # 水坝类型说明
    dam_type_descriptions = {
        'e': "土石坝（embankment dam）：主要用于防止水库蓄水，适用于高山地形（如山谷、峡谷）",
        'g': "重力坝(gravity dam)：靠自身重量抵抗水的压力，适用于各种地形条件",
        'b': "堰坝(barrage dam)：主要用于调节水位和引水，通常较低且宽度大，包含其他类型(other)水坝",
        'a': "拱坝(arch dam)：利用拱形结构将水压力传递给两侧山体，适用于峡谷地形"
    }
    
    # 指标类型说明
    metric_type_descriptions = {
        'count': "数量：表示特定年份特定类型水坝的绝对数量",
        'pct': "百分比：表示特定类型水坝占流域内所有水坝的百分比",
        'change_10to20': "2010-2020变化率：表示2010年到2020年间水坝数量的变化百分比",
        'change_15to20': "2015-2020变化率：表示2015年到2020年间水坝数量的变化百分比"
    }
    
    # 创建汇总报告文件
    summary_file = os.path.join(output_dir, "summary_report.txt")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("流域水坝特征分析汇总报告\n")
        f.write("=" * 100 + "\n\n")
        
        # 写入报告简介
        f.write("报告概述\n")
        f.write("-" * 50 + "\n")
        f.write("本报告分析了不同类型流域中水坝特征的差异。流域按以下五个指标的中位数分为高值组和低值组，\n")
        f.write("然后比较了两组之间在水坝数量、类型分布和变化率方面的差异。显著性水平设定为p<0.05。\n\n")
        
        # 写入指标说明
        f.write("流域分类指标说明\n")
        f.write("-" * 50 + "\n")
        for indicator, description in indicator_descriptions.items():
            f.write(f"{indicator}: {description}\n")
        f.write("\n")
        
        # 写入水坝类型说明
        f.write("水坝类型说明\n")
        f.write("-" * 50 + "\n")
        for code, description in dam_type_descriptions.items():
            f.write(f"{code}: {description}\n")
        f.write("\n")
        
        # 写入指标类型说明
        f.write("分析指标类型说明\n")
        f.write("-" * 50 + "\n")
        for code, description in metric_type_descriptions.items():
            f.write(f"{code}: {description}\n")
        f.write("\n")
        
        # 写入指标命名规则说明
        f.write("指标命名规则\n")
        f.write("-" * 50 + "\n")
        f.write("指标的命名遵循以下格式：\n")
        f.write("1. 年份_水坝类型: 表示该年该类型水坝的数量，例如'2020_g'表示2020年重力坝的数量\n")
        f.write("2. 年份_水坝类型_pct: 表示该类型水坝占总数的百分比，例如'2020_b_pct'表示2020年堰坝占总数的百分比\n")
        f.write("3. change_10to20_水坝类型: 表示2010-2020年间该类型水坝的变化率\n")
        f.write("4. change_15to20_水坝类型: 表示2015-2020年间该类型水坝的变化率\n")
        f.write("5. 年份_total: 表示该年份水坝的总数量\n")
        f.write("6. change_10to20_total和change_15to20_total: 表示总水坝数量的变化率\n\n")
        
        # 分析结果摘要
        f.write("分析结果摘要\n")
        f.write("=" * 100 + "\n\n")
        
        significant_found = False
        
        # 对每个指标进行详细分析
        for indicator, results in all_results.items():
            f.write(f"\n指标: {indicator} ({indicator_descriptions.get(indicator, '无描述')})\n")
            f.write("-" * 100 + "\n")
            
            # 筛选出显著差异的指标
            significant_results = results[results['P_Value'] < 0.05]
            
            if len(significant_results) > 0:
                significant_found = True
                f.write(f"在按{indicator}分组的分析中，发现以下{len(significant_results)}个指标存在显著差异:\n\n")
                
                # 按指标类型整理结果
                counts = []
                percentages = []
                changes_10to20 = []
                changes_15to20 = []
                
                for i, row in significant_results.iterrows():
                    metric = row['Metric']
                    high_mean = row['High_Mean']
                    low_mean = row['Low_Mean']
                    p_value = row['P_Value']
                    
                    # 格式化结果字符串
                    result_str = f"{metric}: 高指标组平均值={high_mean:.2f}, 低指标组平均值={low_mean:.2f}, p值={p_value:.4f}"
                    
                    # 分类存储
                    if metric.endswith('_pct'):
                        percentages.append(result_str)
                    elif metric.startswith('change_10to20'):
                        changes_10to20.append(result_str)
                    elif metric.startswith('change_15to20'):
                        changes_15to20.append(result_str)
                    elif metric.startswith('2020_'):
                        counts.append(result_str)
                
                # 按类别输出结果
                if counts:
                    f.write("【2020年水坝数量】\n")
                    for result in counts:
                        f.write(f"  - {result}\n")
                    f.write("\n")
                
                if percentages:
                    f.write("【2020年水坝类型百分比】\n")
                    for result in percentages:
                        f.write(f"  - {result}\n")
                    f.write("\n")
                
                if changes_10to20:
                    f.write("【2010-2020年变化率】\n")
                    for result in changes_10to20:
                        f.write(f"  - {result}\n")
                    f.write("\n")
                
                if changes_15to20:
                    f.write("【2015-2020年变化率】\n")
                    for result in changes_15to20:
                        f.write(f"  - {result}\n")
                    f.write("\n")
                
                # 添加解释性结论
                f.write("解释说明:\n")
                
                for i, row in significant_results.iterrows():
                    metric = row['Metric']
                    high_mean = row['High_Mean']
                    low_mean = row['Low_Mean']
                    
                    # 根据指标类型提供不同的解释
                    if metric.endswith('_pct'):
                        dam_type = metric.split('_')[1]
                        dam_name = dam_type_descriptions.get(dam_type, dam_type)
                        f.write(f"  - {metric}: 在{indicator}较高的流域中，{dam_name}占比")
                        f.write("更高" if high_mean > low_mean else "更低")
                        f.write(f"（{high_mean:.2f}% vs {low_mean:.2f}%）。\n")
                    
                    elif metric.startswith('change'):
                        period = "2010-2020年" if "10to20" in metric else "2015-2020年"
                        dam_type = metric.split('_')[-1]
                        dam_name = dam_type_descriptions.get(dam_type, dam_type)
                        f.write(f"  - {metric}: 在{indicator}较高的流域中，{period}期间{dam_name}")
                        f.write("增长更快" if high_mean > low_mean else "增长更慢")
                        f.write(f"（{high_mean:.2f}% vs {low_mean:.2f}%）。\n")
                    
                    elif metric.startswith('2020_'):
                        dam_type = metric.split('_')[1]
                        dam_name = dam_type_descriptions.get(dam_type, dam_type)
                        f.write(f"  - {metric}: 在{indicator}较高的流域中，{dam_name}数量")
                        f.write("更多" if high_mean > low_mean else "更少")
                        f.write(f"（{high_mean:.2f} vs {low_mean:.2f}）。\n")
                
                f.write("\n")
            else:
                f.write(f"在按{indicator}分组的分析中，未发现水坝特征有显著差异。这表明水坝的分布和变化可能不受此指标的影响。\n\n")
        
        # 总结
        f.write("\n总体结论\n")
        f.write("=" * 100 + "\n")
        if significant_found:
            f.write("基于以上分析，我们发现不同流域特征确实与水坝的分布和变化存在显著关联。特别是：\n\n")
            
            # 这里可以根据实际结果添加一些总体结论，或者让用户自行解读
            f.write("1. 不同水文特征（如径流量、降水量）的流域在水坝类型分布上存在差异\n")
            f.write("2. 社会经济发展水平（HDI指数）可能影响水坝建设的速度和类型选择\n")
            f.write("3. 水资源条件（如地下水位）与水坝建设策略存在关联\n\n")
            
            f.write("这些发现对于理解水坝建设的区域差异性和预测未来水坝发展趋势具有重要意义。\n")
        else:
            f.write("基于所有分析指标，未发现流域特征与水坝分布和变化之间存在一致的显著关联。\n")
            f.write("这可能说明水坝的建设和分布受到更复杂的因素组合影响，而非单一流域特征决定。\n")
        
        # 添加研究局限性说明
        f.write("\n研究局限性\n")
        f.write("-" * 50 + "\n")
        f.write("1. 本分析仅考虑了流域按指标中位数的二分法，可能掩盖了更复杂的关系\n")
        f.write("2. 未考虑地理位置、政策环境等可能影响水坝建设的其他重要因素\n")
        f.write("3. 变化率计算中，对于初始值为零的情况处理可能影响结果解释\n")
        f.write("4. 分析基于现有数据，可能存在数据不完整或分类不准确的问题\n\n")
        
        # 添加数据来源和分析日期
        f.write("\n数据来源与分析信息\n")
        f.write("-" * 50 + "\n")
        f.write(f"分析日期: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n")
        f.write("流域数据来源: BasinATLAS v10\n")
        f.write("水坝数据来源: Global Dam Dataset v4/v5\n")
    
    logging.info(f"详细汇总报告已保存至 {summary_file}")

def main():
    setup_logging()
    
    # 定义文件路径
    basin_file = r"E:\wyj\project\dam\result1\output\BasinATLAS_v10_lev03_with_dam_stats_v6.shp"
    output_dir = r"E:\wyj\dam\v4\image\result1\analysis_results_v2"
    os.makedirs(output_dir, exist_ok=True)
    
    # 要分析的指标列表
    indicators = ['dis_m3_pyr', 'run_mm_syr', 'gwt_cm_sav', 'pre_mm_syr', 'hdi_ix_sav', 'ari_ix_sav']
    
    # 加载数据
    gdf = load_basin_data(basin_file)
    
    # 计算水坝指标
    gdf = calculate_dam_metrics(gdf)
    
    # 保存中间数据以便检查
    interim_file = os.path.join(output_dir, "basin_with_metrics.csv")
    gdf.drop('geometry', axis=1).to_csv(interim_file, index=False)
    logging.info(f"中间计算结果已保存至 {interim_file}")
    
    # 存储所有分析结果
    all_results = {}
    
    # 按每个指标分析
    for indicator in indicators:
        if indicator in gdf.columns:
            results = analyze_by_indicator(gdf, indicator, output_dir)
            all_results[indicator] = results
        else:
            logging.warning(f"指标 {indicator} 不在数据集中，跳过分析")
    
    # 添加这里 - 为HDI指标创建箱型图
    if 'hdi_ix_sav' in gdf.columns:
        figures_dir = create_boxplots_for_hdi(gdf, output_dir)
        logging.info(f"HDI相关图表已保存至: {figures_dir}")

    # 生成汇总报告
    generate_summary_report(all_results, output_dir)
    
    logging.info("分析完成")

if __name__ == "__main__":
    main()