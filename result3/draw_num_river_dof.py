# Function toimport os
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import MultiLineString, LineString
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import json
from functools import partial
import os
from tqdm import tqdm

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
colors = sns.color_palette("viridis", 10)

# Function to load and prepare data from a GPKG file
def load_river_data(gpkg_file, layer='DOF'):
    """
    Load river data from a GeoPackage file.
    
    Parameters:
    -----------
    gpkg_file : str
        Path to the GeoPackage file.
    layer : str
        Layer name in the GeoPackage file.
        
    Returns:
    --------
    gdf : GeoDataFrame
        Geodataframe containing river data.
    """
    print(f"Loading data from {os.path.basename(gpkg_file)}...")
    
    try:
        # Convert numeric columns that might be strings
        numeric_cols = ['DOF', 'dof_all', 'LENGTH_KM', 'DIS_AV_CMS', 'RIV_ORD', 'VOLUME_TCM', 'DOR', 'SED']

        # 读取数据，排除几何列
        gdf = gpd.read_file(gpkg_file, layer=layer, columns=numeric_cols)
        print(f"Successfully loaded {len(gdf)} river segments")

        # Convert numeric columns that might be strings
        for col in numeric_cols:
            if col in gdf.columns:
                gdf[col] = pd.to_numeric(gdf[col], errors='coerce').fillna(0)
        
        # Make sure GOID is string type for joins
        if 'GOID' in gdf.columns:
            gdf['GOID'] = gdf['GOID'].astype(str)
        
        # Check for invalid geometries
        invalid_geoms = gdf[~gdf.geometry.is_valid]
        if len(invalid_geoms) > 0:
            print(f"Warning: Found {len(invalid_geoms)} invalid geometries. Attempting to fix...")
            gdf.geometry = gdf.geometry.buffer(0)  # Simple fix for some invalid geometries
        
        return gdf
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Function to analyze river fragmentation
def analyze_fragmentation(gdf, gpkg_file=None):
    """
    Analyze river fragmentation based on DOF and dof_all values.
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        Geodataframe containing river data.
    gpkg_file : str
        Path to the GeoPackage file that was analyzed.
        
    Returns:
    --------
    analysis_results : dict
        Dictionary containing analysis results.
    """
    analysis_results = {}
    
    # Check if required columns exist
    required_cols = ['DOF', 'dof_all', 'LENGTH_KM', 'DIS_AV_CMS', 'RIV_ORD']
    missing_cols = [col for col in required_cols if col not in gdf.columns]
    
    if missing_cols:
        print(f"Warning: Missing required columns: {missing_cols}")
        return None
    
    # 1. Count segments by fragmentation status
    # Define fragmentation categories based on dof_all
    gdf['frag_category'] = pd.cut(
        gdf['dof_all'], 
        bins=[-0.1, 0, 25, 50, 75, 100],
        labels=['None (0)', 'Low (1-25)', 'Medium (26-50)', 'High (51-75)', 'Very High (76-100)']
    )
    
    segment_counts = gdf['frag_category'].value_counts().sort_index()
    analysis_results['segment_counts'] = segment_counts
    # print("\n1. River Segment Counts by Fragmentation Category:")
    # print(segment_counts)
    
    # 2. Total length by fragmentation category
    length_by_category = gdf.groupby('frag_category')['LENGTH_KM'].sum()
    analysis_results['length_by_category'] = length_by_category
    # print("\n2. Total River Length (km) by Fragmentation Category:")
    # print(length_by_category)
    
    # 3. Average discharge by fragmentation category
    discharge_by_category = gdf.groupby('frag_category')['DIS_AV_CMS'].mean()
    analysis_results['discharge_by_category'] = discharge_by_category
    # print("\n3. Average Discharge (cms) by Fragmentation Category:")
    # print(discharge_by_category)
    
    # 4. Distribution by river order
    order_by_category = gdf.groupby(['frag_category', 'RIV_ORD']).size().unstack(fill_value=0)
    analysis_results['order_by_category'] = order_by_category
    # print("\n4. River Order Distribution by Fragmentation Category:")
    # print(order_by_category)
    
    # 5. Total volume by fragmentation category (if available)
    if 'VOLUME_TCM' in gdf.columns:
        volume_by_category = gdf.groupby('frag_category')['VOLUME_TCM'].sum()
        analysis_results['volume_by_category'] = volume_by_category
        # print("\n5. Total Volume (cubic km) by Fragmentation Category:")
        # print(volume_by_category)
    
    # 6. Advanced analysis: Fragmentation by continent and basin (if available)
    if 'CONTINENT' in gdf.columns and 'BAS_NAME' in gdf.columns:
        # Continent analysis
        continent_analysis = gdf.groupby('CONTINENT')['dof_all'].agg(['mean', 'median', 'count'])
        analysis_results['continent_analysis'] = continent_analysis
        # print("\n6a. Fragmentation by Continent:")
        # print(continent_analysis)
        basin_names = gdf['BAS_NAME'].unique()
        for basin in basin_names:
            if basin == "" or basin == ":" or basin is None:
                print(f'Warning: {gpkg_file} has invalid basin name: {basin}')

        # Basin analysis (top 10 basins by segment count)
        basin_analysis = gdf.groupby('BAS_NAME').agg({
            'dof_all': ['mean', 'median'],
            'LENGTH_KM': 'sum',
            'GOID': 'count'
        }).sort_values(('GOID', 'count'), ascending=False).head(10)
        analysis_results['basin_analysis'] = basin_analysis
        # print("\n6b. Fragmentation by Top 10 Basins:")
        # print(basin_analysis)
    
    return analysis_results


# Function to visualize analysis results
def visualize_results(analysis_results, output_folder):
    """
    Create visualizations based on analysis results.
    
    Parameters:
    -----------
    analysis_results : dict
        Dictionary containing analysis results.
    output_folder : str
        Folder to save visualizations.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # 1. Segment counts by fragmentation category
    if 'segment_counts' in analysis_results:
        plt.figure(figsize=(12, 8))
        ax = analysis_results['segment_counts'].plot(
            kind='bar', 
            color=colors, 
            alpha=0.8
        )
        plt.title('Number of River Segments by Fragmentation Category')
        plt.xlabel('Fragmentation Category')
        plt.ylabel('Number of Segments')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for i, v in enumerate(analysis_results['segment_counts']):
            ax.text(i, v + 5, f"{v:,}", ha='center', fontweight='bold')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'segment_counts.png'), dpi=300)
        plt.close()
    
    # 2. River length by fragmentation category
    if 'length_by_category' in analysis_results:
        plt.figure(figsize=(12, 8))
        ax = analysis_results['length_by_category'].plot(
            kind='bar', 
            color=colors, 
            alpha=0.8
        )
        plt.title('Total River Length by Fragmentation Category')
        plt.xlabel('Fragmentation Category')
        plt.ylabel('Total Length (km)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for i, v in enumerate(analysis_results['length_by_category']):
            ax.text(i, v + (v*0.02), f"{v:,.0f}", ha='center', fontweight='bold')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'length_by_category.png'), dpi=300)
        plt.close()
    
    # 3. Average discharge by fragmentation category
    if 'discharge_by_category' in analysis_results:
        plt.figure(figsize=(12, 8))
        ax = analysis_results['discharge_by_category'].plot(
            kind='bar', 
            color=colors, 
            alpha=0.8
        )
        plt.title('Average River Discharge by Fragmentation Category')
        plt.xlabel('Fragmentation Category')
        plt.ylabel('Average Discharge (m³/s)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for i, v in enumerate(analysis_results['discharge_by_category']):
            ax.text(i, v + (v*0.02), f"{v:.2f}", ha='center', fontweight='bold')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'discharge_by_category.png'), dpi=300)
        plt.close()
    
    # 4. River order distribution by fragmentation category
    if 'order_by_category' in analysis_results:
        plt.figure(figsize=(14, 10))
        analysis_results['order_by_category'].plot(
            kind='bar', 
            stacked=True, 
            colormap='viridis',
            alpha=0.8
        )
        plt.title('River Order Distribution by Fragmentation Category')
        plt.xlabel('Fragmentation Category')
        plt.ylabel('Number of Segments')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='River Order', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'order_distribution.png'), dpi=300)
        plt.close()
    
    # 5. Fragmentation by continent (if available)
    if 'continent_analysis' in analysis_results:
        plt.figure(figsize=(14, 8))
        continent_data = analysis_results['continent_analysis']
        
        # Sort by mean fragmentation
        continent_data = continent_data.sort_values('mean', ascending=False)
        
        # Plot mean fragmentation by continent
        ax = continent_data['mean'].plot(
            kind='bar', 
            color=colors, 
            alpha=0.8
        )
        plt.title('Average Fragmentation by Continent')
        plt.xlabel('Continent')
        plt.ylabel('Average DOF Score')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on bars
        for i, v in enumerate(continent_data['mean']):
            ax.text(i, v + 1, f"{v:.2f}", ha='center', fontweight='bold')
            
        # Add segment count as text
        for i, count in enumerate(continent_data['count']):
            ax.text(i, 2, f"n={count:,}", ha='center', fontweight='bold', color='white')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'continent_fragmentation.png'), dpi=300)
        plt.close()
    
    # 6. Combined 2x2 visualization
    plt.figure(figsize=(20, 16))
    
    # Plot 1: Segment counts
    if 'segment_counts' in analysis_results:
        plt.subplot(2, 2, 1)
        analysis_results['segment_counts'].plot(
            kind='bar', 
            color=colors, 
            alpha=0.8
        )
        plt.title('Number of River Segments by\nFragmentation Category')
        plt.xlabel('Fragmentation Category')
        plt.ylabel('Number of Segments')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 2: Total length
    if 'length_by_category' in analysis_results:
        plt.subplot(2, 2, 2)
        analysis_results['length_by_category'].plot(
            kind='bar', 
            color=colors, 
            alpha=0.8
        )
        plt.title('Total River Length by\nFragmentation Category')
        plt.xlabel('Fragmentation Category')
        plt.ylabel('Total Length (km)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 3: Average discharge
    if 'discharge_by_category' in analysis_results:
        plt.subplot(2, 2, 3)
        analysis_results['discharge_by_category'].plot(
            kind='bar', 
            color=colors, 
            alpha=0.8
        )
        plt.title('Average River Discharge by\nFragmentation Category')
        plt.xlabel('Fragmentation Category')
        plt.ylabel('Average Discharge (m³/s)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 4: River order distribution
    if 'order_by_category' in analysis_results:
        plt.subplot(2, 2, 4)
        analysis_results['order_by_category'].plot(
            kind='bar', 
            stacked=True, 
            colormap='viridis',
            alpha=0.8
        )
        plt.title('River Order Distribution by\nFragmentation Category')
        plt.xlabel('Fragmentation Category')
        plt.ylabel('Number of Segments')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='River Order', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'combined_analysis.png'), dpi=300)
    plt.close()


def generate_report_all(analysis_results, output_folder, gpkg_file):
    """
    Generate a comprehensive text report summarizing the analysis results.
    
    Parameters:
    -----------
    analysis_results : dict
        Dictionary containing analysis results.
    output_folder : str
        Folder to save the report.
    gpkg_file : str
        Path to the GeoPackage file that was analyzed.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    report_path = os.path.join(output_folder, 'fragmentation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RIVER FRAGMENTATION ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"File analyzed: {os.path.basename(gpkg_file)}\n")
        f.write(f"Date of analysis: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        # Summary of segment counts
        if 'segment_counts' in analysis_results:
            f.write("-"*80 + "\n")
            f.write("1. RIVER SEGMENT COUNTS BY FRAGMENTATION CATEGORY\n")
            f.write("-"*80 + "\n")
            segment_counts = analysis_results['segment_counts']
            total_segments = segment_counts.sum()
            
            for category, count in segment_counts.items():
                percentage = (count / total_segments) * 100
                f.write(f"{category}: {count:,} segments ({percentage:.2f}% of total)\n")
            
            f.write(f"\nTotal number of river segments: {total_segments:,}\n\n")
        
        # Summary of river length
        if 'length_by_category' in analysis_results:
            f.write("-"*80 + "\n")
            f.write("2. RIVER LENGTH BY FRAGMENTATION CATEGORY\n")
            f.write("-"*80 + "\n")
            length_by_category = analysis_results['length_by_category']
            total_length = length_by_category.sum()
            
            for category, length in length_by_category.items():
                percentage = (length / total_length) * 100
                f.write(f"{category}: {length:,.2f} km ({percentage:.2f}% of total length)\n")
            
            f.write(f"\nTotal river length: {total_length:,.2f} km\n\n")
        
        # Summary of river discharge
        if 'discharge_by_category' in analysis_results:
            f.write("-"*80 + "\n")
            f.write("3. AVERAGE RIVER DISCHARGE BY FRAGMENTATION CATEGORY\n")
            f.write("-"*80 + "\n")
            discharge_by_category = analysis_results['discharge_by_category']
            
            for category, discharge in discharge_by_category.items():
                f.write(f"{category}: {discharge:.2f} m³/s average discharge\n")
            
            f.write("\n")
        
        # River order distribution
        if 'order_by_category' in analysis_results:
            f.write("-"*80 + "\n")
            f.write("4. RIVER ORDER DISTRIBUTION BY FRAGMENTATION CATEGORY\n")
            f.write("-"*80 + "\n")
            order_by_category = analysis_results['order_by_category']
            
            f.write(str(order_by_category) + "\n\n")
            
            # Additional interpretation for river order
            f.write("Interpretation of River Order:\n")
            f.write("- Lower order (1-3): Largest mainstem rivers\n")
            f.write("- Mid order (4-5): Medium rivers and tributaries\n")
            f.write("- Higher order (6-7): Small tributaries\n\n")

            # NEW: Add fragmentation analysis by river order
            f.write("Fragmentation Analysis by River Order:\n")
            order_totals = order_by_category.sum(axis=0)
            fragmentation_by_order = {}
            
            for order in order_by_category.columns:
                high_frag_count = order_by_category.loc['High (51-75)', order] + order_by_category.loc['Very High (76-100)', order]
                frag_percentage = (high_frag_count / order_totals[order]) * 100 if order_totals[order] > 0 else 0
                fragmentation_by_order[order] = frag_percentage
                
                f.write(f"- River Order {order}: {frag_percentage:.2f}% highly fragmented (DOF > 50)\n")
            
            # Identify which river orders are most affected by fragmentation
            most_fragmented_orders = sorted(fragmentation_by_order.items(), key=lambda x: x[1], reverse=True)[:3]
            least_fragmented_orders = sorted(fragmentation_by_order.items(), key=lambda x: x[1])[:3]
            
            f.write("\nMost fragmented river orders:\n")
            for order, pct in most_fragmented_orders:
                f.write(f"- Order {order}: {pct:.2f}%\n")
                
            f.write("\nLeast fragmented river orders:\n")
            for order, pct in least_fragmented_orders:
                f.write(f"- Order {order}: {pct:.2f}%\n")
            
            f.write("\n")
        
        # Continent analysis
        if 'continent_analysis' in analysis_results:
            f.write("-"*80 + "\n")
            f.write("5. FRAGMENTATION BY CONTINENT\n")
            f.write("-"*80 + "\n")
            continent_analysis = analysis_results['continent_analysis']
            
            for continent, data in continent_analysis.sort_values('mean', ascending=False).iterrows():
                f.write(f"{continent}:\n")
                f.write(f"  - Mean DOF: {data['mean']:.2f}\n")
                f.write(f"  - Median DOF: {data['median']:.2f}\n")
                f.write(f"  - Number of segments: {data['count']:,}\n\n")
            
            # NEW: Add continent ranking and comparison
            f.write("Continental Fragmentation Ranking (Mean DOF):\n")
            for i, (continent, data) in enumerate(continent_analysis.sort_values('mean', ascending=False).iterrows(), 1):
                f.write(f"{i}. {continent}: {data['mean']:.2f}\n")
            
            # Calculate global average for comparison
            global_mean = continent_analysis['mean'].mean()
            f.write(f"\nGlobal continental average DOF: {global_mean:.2f}\n")
            
            # Identify continents above or below global average
            above_avg = continent_analysis[continent_analysis['mean'] > global_mean].index.tolist()
            below_avg = continent_analysis[continent_analysis['mean'] < global_mean].index.tolist()
            
            f.write(f"\nContinents above global average: {', '.join(above_avg)}\n")
            f.write(f"Continents below global average: {', '.join(below_avg)}\n\n")
        
        # Basin analysis
        if 'basin_analysis' in analysis_results:
            f.write("-"*80 + "\n")
            f.write("6. TOP 10 BASINS BY SEGMENT COUNT\n")
            f.write("-"*80 + "\n")
            basin_analysis = analysis_results['basin_analysis']
            
            # 添加这段代码来检测和打印无名流域
            for basin, data in basin_analysis.iterrows():
                # 检查流域名称是否为空或只是冒号
                if basin == "" or basin == ":" or basin is None:
                    f.write(f"WARNING: Unnamed basin detected in file: {os.path.basename(gpkg_file)}\n")
                    print(f"WARNING: Unnamed basin detected in file: {os.path.basename(gpkg_file)}")
                    # 如果想要将这个信息保存到单独的日志文件
                    with open(os.path.join(output_folder, 'unnamed_basins_log.txt'), 'a') as log:
                        log.write(f"Unnamed basin found in file: {gpkg_file}\n")
                        log.write(f"  - Mean DOF: {data[('dof_all', 'mean')]:.2f}\n")
                        log.write(f"  - Total length: {data[('LENGTH_KM', 'sum')]:,.2f} km\n")
                        log.write(f"  - Number of segments: {data[('GOID', 'count')]:,}\n\n")
    

            for basin, data in basin_analysis.iterrows():
                f.write(f"{basin}:\n")
                f.write(f"  - Mean DOF: {data[('dof_all', 'mean')]:.2f}\n")
                f.write(f"  - Median DOF: {data[('dof_all', 'median')]:.2f}\n")
                f.write(f"  - Total length: {data[('LENGTH_KM', 'sum')]:,.2f} km\n")
                f.write(f"  - Number of segments: {data[('GOID', 'count')]:,}\n\n")
            
            # NEW: More detailed basin analysis
            f.write("-"*80 + "\n")
            f.write("7. DETAILED BASIN FRAGMENTATION ANALYSIS\n")
            f.write("-"*80 + "\n")
            
            # Basin ranking by Mean DOF
            f.write("7.1 Basin Ranking by Mean DOF (Most to Least Fragmented):\n")
            for i, (basin, data) in enumerate(basin_analysis.sort_values(('dof_all', 'mean'), ascending=False).iterrows(), 1):
                f.write(f"{i}. {basin}: {data[('dof_all', 'mean')]:.2f}\n")
            
            f.write("\n")
            
            # Relationships between metrics
            f.write("7.2 Basin Fragmentation, Length and Segment Relationships:\n")
            
            # Calculate correlation between basin size and fragmentation
            corr_size_frag = basin_analysis[('GOID', 'count')].corr(basin_analysis[('dof_all', 'mean')])
            f.write(f"Correlation between basin size (segment count) and mean DOF: {corr_size_frag:.4f}\n")
            
            if abs(corr_size_frag) > 0.5:
                if corr_size_frag > 0:
                    f.write("INSIGHT: Larger basins (by segment count) tend to be more fragmented.\n")
                else:
                    f.write("INSIGHT: Smaller basins (by segment count) tend to be more fragmented.\n")
            else:
                f.write("INSIGHT: There is no strong relationship between basin size and fragmentation level.\n")
            
            # Calculate correlation between basin length and fragmentation
            corr_length_frag = basin_analysis[('LENGTH_KM', 'sum')].corr(basin_analysis[('dof_all', 'mean')])
            f.write(f"Correlation between total basin length and mean DOF: {corr_length_frag:.4f}\n")
            
            if abs(corr_length_frag) > 0.5:
                if corr_length_frag > 0:
                    f.write("INSIGHT: Basins with greater total river length tend to be more fragmented.\n")
                else:
                    f.write("INSIGHT: Basins with lower total river length tend to be more fragmented.\n")
            else:
                f.write("INSIGHT: There is no strong relationship between total basin length and fragmentation level.\n")
            
            # Identify basins with high mean but low median (skewed distribution)
            skewed_basins = []
            for basin, data in basin_analysis.iterrows():
                if data[('dof_all', 'mean')] > 30 and data[('dof_all', 'median')] < 10:
                    skewed_basins.append((basin, data[('dof_all', 'mean')], data[('dof_all', 'median')]))
            
            if skewed_basins:
                f.write("\n7.3 Basins with Skewed Fragmentation Distribution (high mean, low median):\n")
                f.write("These basins likely have a small number of highly fragmented segments that drive up the mean,\n")
                f.write("while the majority of segments remain relatively unfragmented:\n")
                
                for basin, mean, median in skewed_basins:
                    f.write(f"- {basin}: Mean DOF = {mean:.2f}, Median DOF = {median:.2f}\n")
            
            # Basin density calculation (length per segment)
            f.write("\n7.4 Basin Density Analysis (river km per segment):\n")
            density_data = []
            
            for basin, data in basin_analysis.iterrows():
                segment_count = data[('GOID', 'count')]
                total_length = data[('LENGTH_KM', 'sum')]
                density = total_length / segment_count if segment_count > 0 else 0
                density_data.append((basin, density, data[('dof_all', 'mean')]))
            
            # Sort by density
            density_data.sort(key=lambda x: x[1], reverse=True)
            
            for basin, density, mean_dof in density_data:
                f.write(f"- {basin}: {density:.2f} km per segment, Mean DOF = {mean_dof:.2f}\n")
            
            # Calculate correlation between density and fragmentation
            densities = [d[1] for d in density_data]
            mean_dofs = [d[2] for d in density_data]
            density_frag_corr = np.corrcoef(densities, mean_dofs)[0, 1]
            
            f.write(f"\nCorrelation between basin density and mean DOF: {density_frag_corr:.4f}\n")
            
            if abs(density_frag_corr) > 0.5:
                if density_frag_corr > 0:
                    f.write("INSIGHT: Basins with higher river density (longer segments) tend to be more fragmented.\n")
                else:
                    f.write("INSIGHT: Basins with lower river density (shorter segments) tend to be more fragmented.\n")
            else:
                f.write("INSIGHT: There is no strong relationship between basin river density and fragmentation level.\n")
            
            # NEW: Add analysis on basin patterns
            f.write("\n7.5 Geographic Distribution of Fragmentation:\n")
            
            # Group basins by continent if available
            if 'continent_analysis' in analysis_results:
                continents = continent_analysis.index.tolist()
                basin_continents = {}
                
                # This is a simplification - in real code you'd need to assign basins to continents
                # based on actual geographic data
                
                # Example assignment based on common knowledge
                continent_basins = {
                    "North America": ["Mississippi"],
                    "South America": ["Amazon", "Parana", "Orinoco"],
                    "Europe": [],
                    "Africa": ["Congo", "Nile"],
                    "Asia": ["Ganges", "Yangtse", "Yenisey"]
                }
                
                # Analyze continental patterns of basin fragmentation
                f.write("Basin fragmentation patterns by continent:\n")
                
                for continent, basins in continent_basins.items():
                    if not basins:
                        continue
                        
                    basin_dofs = [basin_analysis.loc[basin, ('dof_all', 'mean')] for basin in basins if basin in basin_analysis.index]
                    
                    if basin_dofs:
                        avg_basin_dof = sum(basin_dofs) / len(basin_dofs)
                        f.write(f"- {continent}: Average basin DOF = {avg_basin_dof:.2f}\n")
                        
                        # List basins in order of fragmentation
                        continent_basin_data = [(b, basin_analysis.loc[b, ('dof_all', 'mean')]) 
                                                for b in basins if b in basin_analysis.index]
                        continent_basin_data.sort(key=lambda x: x[1], reverse=True)
                        
                        f.write(f"  Basins by fragmentation: {', '.join([f'{b} ({d:.1f})' for b, d in continent_basin_data])}\n")
        
        # NEW: Add analysis of relationship between discharge and fragmentation
        if 'discharge_by_category' in analysis_results:
            f.write("-"*80 + "\n")
            f.write("8. DISCHARGE AND FRAGMENTATION RELATIONSHIP ANALYSIS\n")
            f.write("-"*80 + "\n")
            
            discharge_by_category = analysis_results['discharge_by_category']
            
            # Calculate average discharge ratio between fragmentation categories
            if 'None (0)' in discharge_by_category and discharge_by_category['None (0)'] > 0:
                none_discharge = discharge_by_category['None (0)']
                
                f.write("Relative Discharge Ratios (compared to unfragmented rivers):\n")
                
                for category, discharge in discharge_by_category.items():
                    if category != 'None (0)':
                        ratio = discharge / none_discharge
                        f.write(f"- {category} rivers have {ratio:.2f}x the average discharge of unfragmented rivers\n")
            
            # Sort categories by discharge
            sorted_by_discharge = discharge_by_category.sort_values(ascending=False)
            
            f.write("\nFragmentation Categories Ranked by Average Discharge:\n")
            for i, (category, discharge) in enumerate(sorted_by_discharge.items(), 1):
                f.write(f"{i}. {category}: {discharge:.2f} m³/s\n")
            
            # Analyze the pattern
            f.write("\nDischarge-Fragmentation Pattern Analysis:\n")
            
            # Check if discharge increases with fragmentation
            is_monotonic = True
            prev_discharge = -1
            for category in ['None (0)', 'Low (1-25)', 'Medium (26-50)', 'High (51-75)', 'Very High (76-100)']:
                if category in discharge_by_category:
                    curr_discharge = discharge_by_category[category]
                    if prev_discharge != -1 and curr_discharge < prev_discharge:
                        is_monotonic = False
                        break
                    prev_discharge = curr_discharge
            
            if is_monotonic:
                f.write("INSIGHT: There is a clear pattern of increasing average discharge with higher fragmentation levels.\n")
                f.write("This suggests that larger rivers (with higher discharge) are more likely to be fragmented,\n")
                f.write("likely due to their greater economic value for hydropower, irrigation, and other water uses.\n")
            else:
                f.write("The relationship between discharge and fragmentation is not strictly monotonic,\n")
                f.write("suggesting more complex factors affect which rivers are fragmented.\n")
            
            # Calculate the ratio of highest to lowest discharge
            highest_discharge = max(discharge_by_category.values)
            lowest_discharge = min(discharge_by_category.values)
            discharge_ratio = highest_discharge / lowest_discharge if lowest_discharge > 0 else float('inf')
            
            f.write(f"\nThe discharge ratio between the highest and lowest fragmentation categories is {discharge_ratio:.2f}x.\n")
            
            if discharge_ratio > 10:
                f.write("This large difference indicates a strong bias toward fragmentation of high-discharge rivers.\n")
            elif discharge_ratio > 5:
                f.write("This moderate difference indicates a preference toward fragmenting higher-discharge rivers.\n")
            else:
                f.write("This relatively small difference suggests fragmentation affects rivers of all discharge levels.\n")
        
        # NEW: Add temporal analysis if data available
        if 'temporal_data' in analysis_results:
            f.write("-"*80 + "\n")
            f.write("9. TEMPORAL FRAGMENTATION TRENDS\n")
            f.write("-"*80 + "\n")
            
            # This is a placeholder - in real code, you would need actual temporal data
            # This shows how you might structure this section
            f.write("NOTE: This section requires temporal data on dam construction dates or changes in fragmentation over time.\n")
            f.write("      Include this section only if such data is available.\n")
        
        # Conclusion
        f.write("="*80 + "\n")
        f.write("CONCLUSION\n")
        f.write("="*80 + "\n\n")
        
        if 'segment_counts' in analysis_results and 'length_by_category' in analysis_results:
            # Calculate percentage of unfragmented rivers
            unfragmented_count = analysis_results['segment_counts'].get('None (0)', 0)
            total_count = analysis_results['segment_counts'].sum()
            unfragmented_length = analysis_results['length_by_category'].get('None (0)', 0)
            total_length = analysis_results['length_by_category'].sum()
            
            unfragmented_pct_count = (unfragmented_count / total_count) * 100 if total_count > 0 else 0
            unfragmented_pct_length = (unfragmented_length / total_length) * 100 if total_length > 0 else 0
            
            f.write(f"This analysis examined {total_count:,} river segments totaling {total_length:,.2f} km in length.\n\n")
            
            f.write(f"Unfragmented rivers (DOF = 0) account for:\n")
            f.write(f"- {unfragmented_count:,} segments ({unfragmented_pct_count:.2f}% of total segments)\n")
            f.write(f"- {unfragmented_length:,.2f} km ({unfragmented_pct_length:.2f}% of total river length)\n\n")
            
            # Calculate highly fragmented rivers
            high_frag_categories = ['High (51-75)', 'Very High (76-100)']
            high_frag_count = sum(analysis_results['segment_counts'].get(cat, 0) for cat in high_frag_categories)
            high_frag_length = sum(analysis_results['length_by_category'].get(cat, 0) for cat in high_frag_categories)
            
            high_frag_pct_count = (high_frag_count / total_count) * 100 if total_count > 0 else 0
            high_frag_pct_length = (high_frag_length / total_length) * 100 if total_length > 0 else 0
            
            f.write(f"Highly fragmented rivers (DOF > 50) account for:\n")
            f.write(f"- {high_frag_count:,} segments ({high_frag_pct_count:.2f}% of total segments)\n")
            f.write(f"- {high_frag_length:,.2f} km ({high_frag_pct_length:.2f}% of total river length)\n\n")
            
            # NEW: Add summary of key insights
            f.write("KEY INSIGHTS:\n\n")
            
            # Summarize discharge-fragmentation relationship
            if 'discharge_by_category' in analysis_results:
                discharge_by_category = analysis_results['discharge_by_category']
                if 'Very High (76-100)' in discharge_by_category and 'None (0)' in discharge_by_category:
                    ratio = discharge_by_category['Very High (76-100)'] / discharge_by_category['None (0)']
                    f.write(f"1. Highly fragmented rivers have an average discharge {ratio:.1f}x greater than unfragmented rivers,\n")
                    f.write("   suggesting a strong bias toward fragmenting high-discharge rivers for water resources.\n\n")
            
            # Summarize continental patterns
            if 'continent_analysis' in analysis_results:
                most_frag_continent = continent_analysis.sort_values('mean', ascending=False).index[0]
                least_frag_continent = continent_analysis.sort_values('mean').index[0]
                
                most_frag_mean = continent_analysis.loc[most_frag_continent, 'mean']
                least_frag_mean = continent_analysis.loc[least_frag_continent, 'mean']
                
                ratio = most_frag_mean / least_frag_mean if least_frag_mean > 0 else float('inf')
                
                f.write(f"2. {most_frag_continent} has the highest fragmentation level (mean DOF: {most_frag_mean:.2f}),\n")
                f.write(f"   while {least_frag_continent} has the lowest (mean DOF: {least_frag_mean:.2f}),\n")
                f.write(f"   a {ratio:.1f}x difference that reflects different development histories and policies.\n\n")
            
            # Summarize basin patterns
            if 'basin_analysis' in analysis_results:
                most_frag_basin = basin_analysis.sort_values(('dof_all', 'mean'), ascending=False).index[0]
                least_frag_basin = basin_analysis.sort_values(('dof_all', 'mean')).index[0]
                
                most_frag_mean = basin_analysis.loc[most_frag_basin, ('dof_all', 'mean')]
                least_frag_mean = basin_analysis.loc[least_frag_basin, ('dof_all', 'mean')]
                
                ratio = most_frag_mean / least_frag_mean if least_frag_mean > 0 else float('inf')
                
                f.write(f"3. Among major river basins, {most_frag_basin} is the most fragmented (mean DOF: {most_frag_mean:.2f}),\n")
                f.write(f"   while {least_frag_basin} is the least fragmented (mean DOF: {least_frag_mean:.2f}).\n\n")
            
            # Add conservation implications
            f.write("CONSERVATION IMPLICATIONS:\n\n")
            
            f.write("1. The {:.2f}% of river length that remains unfragmented represents a critical resource\n".format(unfragmented_pct_length))
            f.write("   for maintaining aquatic biodiversity and ecosystem services.\n\n")
            
            f.write("2. Large rivers with high discharge are disproportionately affected by fragmentation,\n")
            f.write("   impacting migratory species and sediment transport to deltas and coastal zones.\n\n")
            
            f.write("3. Regional differences in fragmentation levels highlight the need for tailored\n")
            f.write("   conservation approaches and international cooperation for transboundary basins.\n\n")
        
        f.write("This report provides a quantitative assessment of river fragmentation patterns based on the DOF (Degree of Fragmentation) metric.\n")
        f.write("The analysis can inform river conservation, dam planning, and ecological restoration efforts.\n")
    
    print(f"Report generated and saved to {report_path}")
    return report_path


# Function to generate a comprehensive report
def generate_report(analysis_results, output_folder, gpkg_file):
    """
    Generate a text report summarizing the analysis results.
    
    Parameters:
    -----------
    analysis_results : dict
        Dictionary containing analysis results.
    output_folder : str
        Folder to save the report.
    gpkg_file : str
        Path to the GeoPackage file that was analyzed.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    report_path = os.path.join(output_folder, 'fragmentation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RIVER FRAGMENTATION ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"File analyzed: {os.path.basename(gpkg_file)}\n")
        f.write(f"Date of analysis: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        # Summary of segment counts
        if 'segment_counts' in analysis_results:
            f.write("-"*80 + "\n")
            f.write("1. RIVER SEGMENT COUNTS BY FRAGMENTATION CATEGORY\n")
            f.write("-"*80 + "\n")
            segment_counts = analysis_results['segment_counts']
            total_segments = segment_counts.sum()
            
            for category, count in segment_counts.items():
                percentage = (count / total_segments) * 100
                f.write(f"{category}: {count:,} segments ({percentage:.2f}% of total)\n")
            
            f.write(f"\nTotal number of river segments: {total_segments:,}\n\n")
        
        # Summary of river length
        if 'length_by_category' in analysis_results:
            f.write("-"*80 + "\n")
            f.write("2. RIVER LENGTH BY FRAGMENTATION CATEGORY\n")
            f.write("-"*80 + "\n")
            length_by_category = analysis_results['length_by_category']
            total_length = length_by_category.sum()
            
            for category, length in length_by_category.items():
                percentage = (length / total_length) * 100
                f.write(f"{category}: {length:,.2f} km ({percentage:.2f}% of total length)\n")
            
            f.write(f"\nTotal river length: {total_length:,.2f} km\n\n")
        
        # Summary of river discharge
        if 'discharge_by_category' in analysis_results:
            f.write("-"*80 + "\n")
            f.write("3. AVERAGE RIVER DISCHARGE BY FRAGMENTATION CATEGORY\n")
            f.write("-"*80 + "\n")
            discharge_by_category = analysis_results['discharge_by_category']
            
            for category, discharge in discharge_by_category.items():
                f.write(f"{category}: {discharge:.2f} m³/s average discharge\n")
            
            f.write("\n")
        
        # River order distribution
        if 'order_by_category' in analysis_results:
            f.write("-"*80 + "\n")
            f.write("4. RIVER ORDER DISTRIBUTION BY FRAGMENTATION CATEGORY\n")
            f.write("-"*80 + "\n")
            order_by_category = analysis_results['order_by_category']
            
            f.write(str(order_by_category) + "\n\n")
            
            # Additional interpretation for river order
            f.write("Interpretation of River Order:\n")
            f.write("- Lower order (1-3): Largest mainstem rivers\n")
            f.write("- Mid order (4-5): Medium rivers and tributaries\n")
            f.write("- Higher order (6-7): Small tributaries\n\n")
        
        # Continent analysis
        if 'continent_analysis' in analysis_results:
            f.write("-"*80 + "\n")
            f.write("5. FRAGMENTATION BY CONTINENT\n")
            f.write("-"*80 + "\n")
            continent_analysis = analysis_results['continent_analysis']
            
            for continent, data in continent_analysis.sort_values('mean', ascending=False).iterrows():
                f.write(f"{continent}:\n")
                f.write(f"  - Mean DOF: {data['mean']:.2f}\n")
                f.write(f"  - Median DOF: {data['median']:.2f}\n")
                f.write(f"  - Number of segments: {data['count']:,}\n\n")
        
        # Basin analysis
        if 'basin_analysis' in analysis_results:
            f.write("-"*80 + "\n")
            f.write("6. TOP 10 BASINS BY SEGMENT COUNT\n")
            f.write("-"*80 + "\n")
            basin_analysis = analysis_results['basin_analysis']
            
            for basin, data in basin_analysis.iterrows():
                f.write(f"{basin}:\n")
                f.write(f"  - Mean DOF: {data[('dof_all', 'mean')]:.2f}\n")
                f.write(f"  - Median DOF: {data[('dof_all', 'median')]:.2f}\n")
                f.write(f"  - Total length: {data[('LENGTH_KM', 'sum')]:,.2f} km\n")
                f.write(f"  - Number of segments: {data[('GOID', 'count')]:,}\n\n")
        
        # Conclusion
        f.write("="*80 + "\n")
        f.write("CONCLUSION\n")
        f.write("="*80 + "\n\n")
        
        if 'segment_counts' in analysis_results and 'length_by_category' in analysis_results:
            # Calculate percentage of unfragmented rivers
            unfragmented_count = analysis_results['segment_counts'].get('None (0)', 0)
            total_count = analysis_results['segment_counts'].sum()
            unfragmented_length = analysis_results['length_by_category'].get('None (0)', 0)
            total_length = analysis_results['length_by_category'].sum()
            
            unfragmented_pct_count = (unfragmented_count / total_count) * 100 if total_count > 0 else 0
            unfragmented_pct_length = (unfragmented_length / total_length) * 100 if total_length > 0 else 0
            
            f.write(f"This analysis examined {total_count:,} river segments totaling {total_length:,.2f} km in length.\n\n")
            
            f.write(f"Unfragmented rivers (DOF = 0) account for:\n")
            f.write(f"- {unfragmented_count:,} segments ({unfragmented_pct_count:.2f}% of total segments)\n")
            f.write(f"- {unfragmented_length:,.2f} km ({unfragmented_pct_length:.2f}% of total river length)\n\n")
            
            # Calculate highly fragmented rivers
            high_frag_categories = ['High (51-75)', 'Very High (76-100)']
            high_frag_count = sum(analysis_results['segment_counts'].get(cat, 0) for cat in high_frag_categories)
            high_frag_length = sum(analysis_results['length_by_category'].get(cat, 0) for cat in high_frag_categories)
            
            high_frag_pct_count = (high_frag_count / total_count) * 100 if total_count > 0 else 0
            high_frag_pct_length = (high_frag_length / total_length) * 100 if total_length > 0 else 0
            
            f.write(f"Highly fragmented rivers (DOF > 50) account for:\n")
            f.write(f"- {high_frag_count:,} segments ({high_frag_pct_count:.2f}% of total segments)\n")
            f.write(f"- {high_frag_length:,.2f} km ({high_frag_pct_length:.2f}% of total river length)\n\n")
        
        f.write("This report provides a quantitative assessment of river fragmentation patterns based on the DOF (Degree of Fragmentation) metric.\n")
        f.write("The analysis can inform river conservation, dam planning, and ecological restoration efforts.\n")
    
    print(f"Report generated and saved to {report_path}")
    return report_path

# Function to process a single GPKG file and return results
def process_single_file(gpkg_file, output_folder):
    """
    Process a single GPKG file and return the analysis results.
    This function is designed to be used with multiprocessing.
    
    Parameters:
    -----------
    gpkg_file : str
        Path to the GPKG file.
    output_folder : str
        Folder to save analysis results.
        
    Returns:
    --------
    result_dict : dict
        Dictionary containing analysis results and metadata.
    """
    start_time = time.time()
    file_base_name = os.path.splitext(os.path.basename(gpkg_file))[0]
    file_output_folder = os.path.join(output_folder, file_base_name)
    os.makedirs(file_output_folder, exist_ok=True)
    
    try:
        print(f"Processing {file_base_name}...")
        
        # Load the data
        gdf = load_river_data(gpkg_file)
        
        if gdf is None:
            print(f"Failed to load data from {file_base_name}. Skipping.")
            return {
                'filename': file_base_name,
                'path': gpkg_file,
                'success': False,
                'error': 'Failed to load data',
                'processing_time': time.time() - start_time
            }
        
        # Add file identifier column
        gdf['source_file'] = file_base_name
        
        # Perform analysis
        analysis_results = analyze_fragmentation(gdf, gpkg_file)
        
        if analysis_results is None:
            print(f"Failed to analyze data from {file_base_name}. Skipping.")
            return {
                'filename': file_base_name,
                'path': gpkg_file,
                'success': False,
                'error': 'Failed to analyze data',
                'processing_time': time.time() - start_time
            }
        
        # Generate visualizations and report
        # visualize_results(analysis_results, file_output_folder)
        report_path = generate_report(analysis_results, file_output_folder, gpkg_file)
        
        # Create a serializable version of the results for multiprocessing
        # (pandas Series and DataFrames need to be converted)
        serializable_results = {}
        for key, value in analysis_results.items():
            if isinstance(value, pd.Series):
                serializable_results[key] = value.to_dict()
            elif isinstance(value, pd.DataFrame):
                serializable_results[key] = value.to_dict()
            else:
                serializable_results[key] = value
    

        # Return results
        result_dict = {
            'filename': file_base_name,
            'path': gpkg_file,
            'success': True,
            'results': serializable_results,
            'report_path': report_path,
            'processing_time': time.time() - start_time,
            'gdf': gdf  # Include the geodataframe for combined analysis
        }
        
        print(f"Completed analysis of {file_base_name} in {time.time() - start_time:.2f} seconds")
        return result_dict
        
    except Exception as e:
        print(f"Error processing {file_base_name}: {str(e)}")
        return {
            'filename': file_base_name,
            'path': gpkg_file,
            'success': False,
            'error': str(e),
            'processing_time': time.time() - start_time
        }

# Function to process multiple GPKG files in a directory using multiprocessing
def process_directory_parallel(directory_path, output_folder=None, max_workers=None):
    """
    Process all GPKG files in a directory using multiprocessing.
    
    Parameters:
    -----------
    directory_path : str
        Path to the directory containing GPKG files.
    output_folder : str, optional
        Folder to save analysis results. If None, a subfolder will be created.
    max_workers : int, optional
        Maximum number of worker processes. If None, uses CPU count.
    
    Returns:
    --------
    all_results : dict
        Dictionary containing results from all files.
    """
    start_time = time.time()
    
    # Set output folder if not provided
    if output_folder is None:
        output_folder = os.path.join(directory_path, "fragmentation_analysis_results_test")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all GPKG files in the directory
    gpkg_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                 if f.lower().endswith('.gpkg')]
    
    # gpkg_files = gpkg_files[:2]

    if not gpkg_files:
        print(f"No GPKG files found in {directory_path}")
        return None
    
    # Determine number of workers (default to CPU count if not specified)
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    print(f"Found {len(gpkg_files)} GPKG files to process using {max_workers} workers")
    
    # Process files in parallel
    results = []
    successful_gdfs = []
    
    # Create a partial function with fixed output_folder parameter
    process_func = partial(process_single_file, output_folder=output_folder)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_func, gpkg_file): gpkg_file for gpkg_file in gpkg_files}
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_file), total=len(gpkg_files), desc="Processing files"):
            gpkg_file = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                
                # If successful and contains GeoDataFrame, add to list for combined analysis
                if result['success'] and 'gdf' in result:
                    successful_gdfs.append(result['gdf'])
                    # Remove GeoDataFrame from result to avoid memory issues
                    del result['gdf']
            except Exception as e:
                print(f"Error processing {os.path.basename(gpkg_file)}: {str(e)}")
                results.append({
                    'filename': os.path.basename(gpkg_file),
                    'path': gpkg_file,
                    'success': False,
                    'error': str(e)
                })
    
    # Organize results
    all_results = {
        'files': [r for r in results if 'filename' in r],
        'combined_results': None,
        'processing_time': time.time() - start_time
    }
    
    # Perform combined analysis if multiple GeoDataFrames were successfully processed
    if len(successful_gdfs) > 1:
        print(f"\nPerforming combined analysis of {len(successful_gdfs)} successfully processed files...")
        
        # Concatenate all dataframes
        combined_output_folder = os.path.join(output_folder, "combined_analysis")
        os.makedirs(combined_output_folder, exist_ok=True)
        
        # try:
        combined_gdf = pd.concat(successful_gdfs, ignore_index=True)
        
        # Analyze the combined data
        combined_results = analyze_fragmentation(combined_gdf)
        
        # Convert pandas objects to dictionaries for serialization
        serializable_combined = {}
        for key, value in combined_results.items():
            if isinstance(value, pd.Series):
                serializable_combined[key] = value.to_dict()
            elif isinstance(value, pd.DataFrame):
                serializable_combined[key] = value.to_dict()
            else:
                serializable_combined[key] = value
        
        all_results['combined_results'] = serializable_combined
        
        # Generate visualizations and report for combined data
        visualize_results(combined_results, combined_output_folder)
        report_path = generate_report_all(combined_results, combined_output_folder, "All GPKG Files")
        
        # Create comparison reports and visualizations
        create_comparison_report(all_results, os.path.join(output_folder, "comparison_summary.txt"))
        create_comparison_visualizations(all_results, combined_output_folder)
        
        print(f"Combined analysis completed. Results saved to {combined_output_folder}")
        # except Exception as e:
        #     print(f"Error in combined analysis: {str(e)}")
    
    # Save results summary as JSON for future reference
    try:
        with open(os.path.join(output_folder, "analysis_summary.json"), 'w') as f:
            # Create a simpler version without complex objects
            simple_summary = {
                'total_files': len(gpkg_files),
                'successful': sum(1 for r in results if r.get('success', False)),
                'failed': sum(1 for r in results if not r.get('success', False)),
                'total_processing_time': all_results['processing_time'],
                'files': [{
                    'filename': r['filename'],
                    'success': r.get('success', False),
                    'processing_time': r.get('processing_time', None),
                    'error': r.get('error', None) if not r.get('success', False) else None
                } for r in results if 'filename' in r]
            }
            json.dump(simple_summary, f, indent=2)
    except Exception as e:
        print(f"Error saving analysis summary: {str(e)}")
    
    print(f"\nAll analyses completed in {time.time() - start_time:.2f} seconds.")
    print(f"Results saved to {output_folder}")
    
    return all_results

# Function to create a comparison report across files
def create_comparison_report(all_results, output_path):
    """
    Create a report comparing results across multiple files.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary containing results from all files.
    output_path : str
        Path to save the comparison report.
    """
    file_results = all_results['files']
    successful_files = [f for f in file_results if f.get('success', False)]
    
    # Function to convert dictionary to Series if needed
    def get_series(result_dict, key):
        if key in result_dict:
            data = result_dict[key]
            if isinstance(data, dict):
                return pd.Series(data)
            return data
        return None
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RIVER FRAGMENTATION COMPARISON ACROSS FILES\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Number of files analyzed: {len(file_results)}\n")
        f.write(f"Successfully processed files: {len(successful_files)}/{len(file_results)}\n")
        f.write(f"Date of analysis: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        if 'processing_time' in all_results:
            f.write(f"Total processing time: {all_results['processing_time']:.2f} seconds\n\n")
        
        # Performance summary
        f.write("-"*80 + "\n")
        f.write("PROCESSING PERFORMANCE\n")
        f.write("-"*80 + "\n\n")
        
        f.write(f"{'File':<30} {'Status':<10} {'Processing Time (s)':<20}\n")
        f.write("-"*65 + "\n")
        
        for file_data in file_results:
            filename = file_data['filename']
            status = "Success" if file_data.get('success', False) else "Failed"
            proc_time = file_data.get('processing_time', "N/A")
            proc_time_str = f"{proc_time:.2f}" if isinstance(proc_time, (int, float)) else proc_time
            
            f.write(f"{filename[:30]:<30} {status:<10} {proc_time_str:<20}\n")
            
            # If failed, include error message
            if not file_data.get('success', False) and 'error' in file_data:
                f.write(f"  Error: {file_data['error']}\n")
        
        f.write("\n")
        
        # Compare unfragmented river percentages
        f.write("-"*80 + "\n")
        f.write("COMPARISON OF UNFRAGMENTED RIVERS (DOF = 0)\n")
        f.write("-"*80 + "\n\n")
        
        f.write(f"{'File':<30} {'Segments':<10} {'% of Total':<15} {'Length (km)':<15} {'% of Total':<15}\n")
        f.write("-"*85 + "\n")
        
        for file_data in successful_files:
            filename = file_data['filename']
            results = file_data['results']
            
            # print(file_data['results'])

            # Get segment counts and length data (handle both Series and dict formats)
            segment_counts = get_series(results, 'segment_counts')
            length_by_category = get_series(results, 'length_by_category')

            # print(f'segment_counts: {segment_counts}')

            if segment_counts is not None and length_by_category is not None:
                # Calculate unfragmented percentages
                unfragmented_count = segment_counts.get('None (0)', 0)
                total_count = sum(segment_counts.values)
                unfragmented_length = length_by_category.get('None (0)', 0)
                total_length = sum(length_by_category.values)
                
                unfragmented_pct_count = (unfragmented_count / total_count) * 100 if total_count > 0 else 0
                unfragmented_pct_length = (unfragmented_length / total_length) * 100 if total_length > 0 else 0
                
                f.write(f"{filename[:30]:<30} {unfragmented_count:<10,d} {unfragmented_pct_count:<15.2f} {unfragmented_length:<15,.2f} {unfragmented_pct_length:<15.2f}\n")
        
        # Compare highly fragmented river percentages
        f.write("\n\n")
        f.write("-"*80 + "\n")
        f.write("COMPARISON OF HIGHLY FRAGMENTED RIVERS (DOF > 50)\n")
        f.write("-"*80 + "\n\n")
        
        f.write(f"{'File':<30} {'Segments':<10} {'% of Total':<15} {'Length (km)':<15} {'% of Total':<15}\n")
        f.write("-"*85 + "\n")
        
        for file_data in successful_files:
            filename = file_data['filename']
            results = file_data['results']
            
            # Get segment counts and length data
            segment_counts = get_series(results, 'segment_counts')
            length_by_category = get_series(results, 'length_by_category')
            
            if segment_counts is not None and length_by_category is not None:
                # Calculate highly fragmented percentages
                high_frag_categories = ['High (51-75)', 'Very High (76-100)']
                high_frag_count = sum(segment_counts.get(cat, 0) for cat in high_frag_categories)
                total_count = sum(segment_counts.values)
                high_frag_length = sum(length_by_category.get(cat, 0) for cat in high_frag_categories)
                total_length = sum(length_by_category.values)
                
                high_frag_pct_count = (high_frag_count / total_count) * 100 if total_count > 0 else 0
                high_frag_pct_length = (high_frag_length / total_length) * 100 if total_length > 0 else 0
                
                f.write(f"{filename[:30]:<30} {high_frag_count:<10,d} {high_frag_pct_count:<15.2f} {high_frag_length:<15,.2f} {high_frag_pct_length:<15.2f}\n")
        
        # Summary stats
        f.write("\n\n")
        f.write("-"*80 + "\n")
        f.write("SUMMARY STATISTICS ACROSS ALL FILES\n")
        f.write("-"*80 + "\n\n")
        
        if 'combined_results' in all_results and all_results['combined_results'] is not None:
            combined = all_results['combined_results']
            
            # Get segment counts
            segment_counts = get_series(combined, 'segment_counts')
            if segment_counts is not None:
                total_segments = sum(segment_counts.values)
                f.write(f"Total river segments across all files: {total_segments:,}\n")
            
                # Get length data
                length_by_category = get_series(combined, 'length_by_category')
                if length_by_category is not None:
                    total_length = sum(length_by_category.values)
                    f.write(f"Total river length across all files: {total_length:,.2f} km\n")
                
                # Fragmentation breakdown
                f.write("\nFragmentation breakdown across all files:\n")
                for category, count in segment_counts.items():
                    percentage = (count / total_segments) * 100
                    f.write(f"- {category}: {count:,} segments ({percentage:.2f}%)\n")
    
    print(f"Comparison report saved to {output_path}")

# Function to update the create_comparison_visualizations function to handle serialized data
def create_comparison_visualizations(all_results, output_folder):
    """
    Create visualizations comparing results across files.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary containing results from all files.
    output_folder : str
        Folder to save the comparison visualizations.
    """
    os.makedirs(output_folder, exist_ok=True)
    file_results = all_results['files']
    
    # Don't create comparisons if only one file was analyzed
    if len(file_results) <= 1:
        return
    
    # Function to convert dictionary to Series if needed
    def get_series(result_dict, key):
        if key in result_dict:
            data = result_dict[key]
            if isinstance(data, dict):
                return pd.Series(data)
            # Check if it's a numpy array
            elif isinstance(data, np.ndarray):
                return pd.Series(data)
            return data
        return None
    
    # 1. Compare unfragmented river percentages across files
    unfrag_pcts = []
    filenames = []
    
    for file_data in file_results:
        if not file_data.get('success', False):
            continue
            
        filename = os.path.splitext(file_data['filename'])[0]
        results = file_data['results']
        
        # Get segment counts and length data (handle both Series and dict formats)
        segment_counts = get_series(results, 'segment_counts')
        length_by_category = get_series(results, 'length_by_category')
        
        if segment_counts is not None and length_by_category is not None:
            # Calculate unfragmented percentages
            unfragmented_count = segment_counts.get('None (0)', 0)
            total_count = sum(segment_counts.values)
            unfragmented_length = length_by_category.get('None (0)', 0)
            total_length = sum(length_by_category.values)
            
            unfragmented_pct_count = (unfragmented_count / total_count) * 100 if total_count > 0 else 0
            unfragmented_pct_length = (unfragmented_length / total_length) * 100 if total_length > 0 else 0
            
            unfrag_pcts.append([unfragmented_pct_count, unfragmented_pct_length])
            filenames.append(filename)
    
    if unfrag_pcts:
        try:
            # Verify that unfrag_pcts contains valid numeric data
            for i, data in enumerate(unfrag_pcts):
                if not isinstance(data, list) or len(data) != 2:
                    print(f"Warning: Invalid data structure for file {filenames[i]}, skipping this entry")
                    # Remove invalid entries
                    unfrag_pcts[i] = [0, 0]  # Replace with zeros or remove completely
            
            unfrag_pcts = np.array(unfrag_pcts, dtype=float)
        
            # Continue with your visualization code...
        except Exception as e:
            print(f"Error converting data to numpy array: {str(e)}")
            # Provide a fallback visualization or skip this visualization

        unfrag_pcts = np.array(unfrag_pcts)
        
        plt.figure(figsize=(14, 8))
        bar_width = 0.35
        x = np.arange(len(filenames))
        
        plt.bar(x - bar_width/2, unfrag_pcts[:, 0], bar_width, label='% of Segments', color='steelblue')
        plt.bar(x + bar_width/2, unfrag_pcts[:, 1], bar_width, label='% of Total Length', color='seagreen')
        
        plt.xlabel('File')
        plt.ylabel('Percentage of Unfragmented Rivers (DOF = 0)')
        plt.title('Comparison of Unfragmented Rivers Across Files')
        plt.xticks(x, filenames, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, v in enumerate(unfrag_pcts[:, 0]):
            plt.text(i - bar_width/2, v + 1, f"{v:.1f}%", ha='center', fontsize=9)
        
        for i, v in enumerate(unfrag_pcts[:, 1]):
            plt.text(i + bar_width/2, v + 1, f"{v:.1f}%", ha='center', fontsize=9)
        
        plt.savefig(os.path.join(output_folder, 'unfragmented_comparison.png'), dpi=300)
        plt.close()
    
    # 2. Compare fragmentation category distribution across files
    frag_categories = ['None (0)', 'Low (1-25)', 'Medium (26-50)', 'High (51-75)', 'Very High (76-100)']
    
    # For segment counts
    frag_data_segments = []
    
    for file_data in file_results:
        if not file_data.get('success', False):
            continue
            
        filename = os.path.splitext(file_data['filename'])[0]
        results = file_data['results']
        
        segment_counts = get_series(results, 'segment_counts')
        
        if segment_counts is not None:
            total_count = sum(segment_counts.values)
            
            # Calculate percentage for each category
            pcts = []
            for cat in frag_categories:
                count = segment_counts.get(cat, 0)
                pct = (count / total_count) * 100 if total_count > 0 else 0
                pcts.append(pct)
            
            frag_data_segments.append(pcts)
        else:
            # Skip this file if no segment counts data
            filenames.remove(filename) if filename in filenames else None
    
    if frag_data_segments and filenames:
        frag_data_segments = np.array(frag_data_segments)
        
        plt.figure(figsize=(14, 10))
        
        # Create stacked bar chart
        bottom = np.zeros(len(filenames))
        
        for i, cat in enumerate(frag_categories):
            plt.bar(filenames, frag_data_segments[:, i], bottom=bottom, label=cat)
            bottom += frag_data_segments[:, i]
        
        plt.xlabel('File')
        plt.ylabel('Percentage of Segments')
        plt.title('Fragmentation Category Distribution (Segments) Across Files')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Fragmentation Category')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.savefig(os.path.join(output_folder, 'fragmentation_segments_comparison.png'), dpi=300)
        plt.close()
    
    # 3. Compare mean discharge by fragmentation category across files
    if len(file_results) >= 2:
        # Create a dictionary to store mean discharge values by category and file
        discharge_data = {}
        
        for file_data in file_results:
            if not file_data.get('success', False):
                continue
                
            filename = os.path.splitext(file_data['filename'])[0]
            results = file_data['results']
            
            discharge_by_category = get_series(results, 'discharge_by_category')
            
            if discharge_by_category is not None:
                discharge_data[filename] = discharge_by_category
        
        if discharge_data:
            # Convert to DataFrame for easier plotting
            discharge_df = pd.DataFrame(discharge_data)
            
            # Plot
            plt.figure(figsize=(14, 8))
            discharge_df.plot(kind='bar', rot=45)
            plt.xlabel('Fragmentation Category')
            plt.ylabel('Average Discharge (m³/s)')
            plt.title('Average Discharge by Fragmentation Category Across Files')
            plt.legend(title='File')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_folder, 'discharge_comparison.png'), dpi=300)
            plt.close()

# Example usage
if __name__ == "__main__":
    # Replace with your directory containing GPKG files
    directory_path = "/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/dof_result_2015"
    
    # Optional: specify number of worker processes (defaults to CPU count if not specified)
    max_workers = None  # Set to a number to limit CPU usage
    
    # Process all files in parallel
    process_directory_parallel(directory_path, max_workers=max_workers)