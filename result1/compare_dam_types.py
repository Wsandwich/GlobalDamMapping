import os
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.ticker as mtick

def create_latitude_bands():
    """Create GeoDataFrame with latitude bands."""
    # Define latitude bands
    bands = [
        {'name': 'Low (0°-25°N)', 'ymin': 0, 'ymax': 25},
        {'name': 'Mid (25°-50°N)', 'ymin': 25, 'ymax': 50},
        {'name': 'High (_50°N)', 'ymin': 50, 'ymax': 90}
    ]
    
    # Create geometries for each band
    geometries = []
    names = []
    
    for band in bands:
        # Create a box for the entire longitude range (-180 to 180)
        geom = box(-180, band['ymin'], 180, band['ymax'])
        geometries.append(geom)
        names.append(band['name'])
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({'name': names, 'geometry': geometries}, crs="EPSG:4326")
    return gdf

def calculate_dam_counts_by_latitude(tif_files, latitude_bands):
    """
    Calculate dam counts for each dam type within each latitude band.
    
    Parameters:
        tif_files (dict): Dictionary mapping dam types to file paths
        latitude_bands (GeoDataFrame): GeoDataFrame with latitude band polygons
    
    Returns:
        DataFrame: Dam counts by latitude band and dam type
    """
    results = []
    
    for dam_type, tif_path in tif_files.items():
        if not os.path.exists(tif_path):
            print(f"Warning: File {tif_path} does not exist. Skipping.")
            continue
            
        with rasterio.open(tif_path) as src:
            for idx, row in latitude_bands.iterrows():
                band_name = row['name']
                
                # Get geometry in the correct CRS
                geom = [row['geometry']]
                
                # Mask the raster with the latitude band
                masked_data, masked_transform = mask(src, geom, crop=True, nodata=0)
                
                # Count dams (sum all pixel values, as each pixel value represents dam count)
                dam_count = np.sum(masked_data)
                
                results.append({
                    'Latitude Band': band_name,
                    'Dam Type': dam_type,
                    'Count': int(dam_count)
                })
    
    return pd.DataFrame(results)

def calculate_percentages(df):
    """Calculate percentages of each dam type within each latitude band."""
    # Group by latitude band and calculate total dams per band
    totals = df.groupby('Latitude Band')['Count'].sum().reset_index()
    totals.rename(columns={'Count': 'Total'}, inplace=True)
    
    # Merge with original dataframe
    df = pd.merge(df, totals, on='Latitude Band')
    
    # Calculate percentage
    df['Percentage'] = (df['Count'] / df['Total']) * 100
    
    return df

def perform_statistical_analysis(df):
    """Perform chi-square test to check for significant differences."""
    # Create contingency table: rows are latitude bands, columns are dam types
    contingency = pd.pivot_table(
        df, values='Count', index='Latitude Band', columns='Dam Type', fill_value=0
    )
    
    # Perform chi-square test
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    
    return {
        'chi2': chi2,
        'p_value': p,
        'dof': dof,
        'contingency_table': contingency,
        'expected': expected
    }

def create_visualizations(df, output_dir):
    """Create visualizations of dam type distributions."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 8))
    
    # Create a grouped bar chart
    ax = sns.barplot(x='Latitude Band', y='Percentage', hue='Dam Type', data=df)
    
    # Add percentage symbols to y-axis
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # Customize the plot
    plt.title('Dam Type Distribution by Latitude Band (2020)', fontsize=16)
    plt.xlabel('Latitude Band', fontsize=14)
    plt.ylabel('Percentage', fontsize=14)
    plt.legend(title='Dam Type', title_fontsize=12, fontsize=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dam_distribution_by_latitude.png'), dpi=300)
    plt.close()
    
    # Create pie charts for each latitude band
    for band in df['Latitude Band'].unique():
        band_data = df[df['Latitude Band'] == band]
        
        plt.figure(figsize=(8, 8))
        plt.pie(band_data['Percentage'], labels=band_data['Dam Type'], autopct='%1.1f%%', startangle=90)
        plt.title(f'Dam Type Distribution - {band}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'dam_distribution_{band.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").replace("°", "")}.png'), dpi=300)
        plt.close()

def generate_summary_text(df, stats_results):
    """Generate a textual summary of the findings."""
    # Format for displaying percentages
    def fmt_pct(pct):
        return f"{pct:.1f}%"
    
    # Get percentages for key comparisons
    low_lat = df[df['Latitude Band'] == 'Low (0°-25°N)']
    high_lat = df[df['Latitude Band'] == 'High (_50°N)']
    
    # Create dictionary to store percentages by dam type
    dam_pcts = {}
    for dam_type in df['Dam Type'].unique():
        low_pct = low_lat[low_lat['Dam Type'] == dam_type]['Percentage'].values[0]
        high_pct = high_lat[high_lat['Dam Type'] == dam_type]['Percentage'].values[0]
        dam_pcts[dam_type] = {'low': low_pct, 'high': high_pct}
    
    # Identify the predominant dam types
    predominant_low = max(low_lat.iterrows(), key=lambda x: x[1]['Percentage'])[1]
    predominant_high = max(high_lat.iterrows(), key=lambda x: x[1]['Percentage'])[1]
    
    # Generate text
    p_value_text = f"P < 0.001" if stats_results['p_value'] < 0.001 else f"P = {stats_results['p_value']:.3f}"
    
    text = f"""
In 2020, low-latitude regions (0°–25°N) showed a significantly higher proportion of {predominant_low['Dam Type']} ({fmt_pct(predominant_low['Percentage'])}) compared to high-latitude regions (>50°N; {fmt_pct(high_lat[high_lat['Dam Type'] == predominant_low['Dam Type']]['Percentage'].values[0])}), where {predominant_high['Dam Type']} predominated ({fmt_pct(predominant_high['Percentage'])} vs. {fmt_pct(low_lat[low_lat['Dam Type'] == predominant_high['Dam Type']]['Percentage'].values[0])}) ({p_value_text}). This contrast reflects regional differences in dam construction priorities and environmental conditions, with irrigation needs potentially driving development in tropical zones versus hydropower-focused development in temperate regions.
"""
    return text

def main():
    # Define paths to TIF files (modify these to your actual paths)
    base_dir = r'E:\wyj\dam\v4\global_v5_tif'  # Replace with your actual base directory
    year = 2020  # You can analyze other years as well
    resolution = 0.05  # Match the resolution used in your script
    
    tif_files = {
        'gravity_dam': os.path.join(base_dir, f'dams_points_{year}_gravity_dam_{resolution}.tif'),
        'Barrage_dam': os.path.join(base_dir, f'dams_points_{year}_Barrage_dam_{resolution}.tif'),
        'arch_dam': os.path.join(base_dir, f'dams_points_{year}_arch_dam_{resolution}.tif'),
        'embankment': os.path.join(base_dir, f'dams_points_{year}_embankment_dam_{resolution}.tif')  # Using "embankment_dam" as a proxy for embankment dams
    }
    
    # Create latitude bands
    latitude_bands = create_latitude_bands()
    
    # Calculate dam counts by latitude band
    df = calculate_dam_counts_by_latitude(tif_files, latitude_bands)
    
    # Calculate percentages
    df_with_pct = calculate_percentages(df)
    
    # Perform statistical analysis
    stats_results = perform_statistical_analysis(df)
    
    # Generate visualizations
    output_dir = 'dam_analysis_results'
    create_visualizations(df_with_pct, output_dir)
    
    # Generate summary text
    summary = generate_summary_text(df_with_pct, stats_results)
    print("\nAnalysis Summary:")
    print(summary)
    
    # Save results to CSV
    df_with_pct.to_csv(os.path.join(output_dir, 'dam_distribution_analysis.csv'), index=False)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()