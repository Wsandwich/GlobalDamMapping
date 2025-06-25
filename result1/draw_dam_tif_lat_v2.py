import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import rasterio
import pandas as pd
import os
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Define the years to process
years = [2010, 2015, 2020]

# Define color scheme
colors = ['#FED59F', '#FDBA83', '#FB8C57']

# Define GeoTIFF file path template
tif_file_template = 'E:/wyj/dam/v4/global_v5_tif/dams_points_{year}_0.05.tif'

# Create dictionary to store data for each year
data_per_year = {}

# Define latitude band width
latitude_band_width = 0.05

# Process data for each year
for idx, year in enumerate(years):
    tif_file_path = tif_file_template.format(year=year)
    if not os.path.exists(tif_file_path):
        print(f"File does not exist: {tif_file_path}")
        continue
    
    with rasterio.open(tif_file_path) as src:
        raster = src.read(1)  # Read first band
        transform = src.transform
        crs = src.crs
        # Get raster bounds
        minx, miny, maxx, maxy = src.bounds
        # Get pixel size
        pixel_size_x = transform.a
        pixel_size_y = -transform.e  # transform.e is usually negative

    # Generate lat/lon coordinates
    new_width = raster.shape[1]
    new_height = raster.shape[0]
    x = np.linspace(minx, maxx, new_width)
    y = np.linspace(maxy, miny, new_height)
    X, Y = np.meshgrid(x, y)

    # Create DataFrame to store latitude and dam count for each pixel
    latitudes = Y.flatten()
    dam_counts = raster.flatten()

    df = pd.DataFrame({
        'latitude': latitudes,
        'dam_count': dam_counts
    })

    # Remove pixels with zero dams
    df = df[df['dam_count'] > 0]

    # Filter out values outside latitude range
    df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90)]

    # Generate bins
    bins = np.arange(-90, 90 + latitude_band_width, latitude_band_width)
    # Generate labels using the lower bound of each band, ensuring float precision
    labels = np.round(bins[:-1], decimals=5)

    # Create latitude bands using pandas.cut
    df['lat_band'] = pd.cut(
        df['latitude'],
        bins=bins,  # From -90 to 90 degrees, each band is latitude_band_width wide
        right=False,  # Left-closed, right-open interval [a, b)
        include_lowest=True,
        labels=labels  # Use the lower bound of each band as the label
    )

    # Verify labels and bins match in length
    assert len(labels) == len(bins) - 1, "Number of labels must be one less than number of bins"

    # Sum dam counts in each latitude band
    count_per_lat_band = df.groupby('lat_band')['dam_count'].sum().reset_index()

    # Convert lat_band to numeric type
    count_per_lat_band['lat_band'] = count_per_lat_band['lat_band'].astype(float)

    # Ensure latitude bands are sorted
    count_per_lat_band = count_per_lat_band.sort_values('lat_band')

    # Store in dictionary
    data_per_year[year] = count_per_lat_band

# Merge data from all years
merged_df = pd.DataFrame({'lat_band': data_per_year[years[0]]['lat_band']})

for year in years:
    if year in data_per_year:
        merged_df = merged_df.merge(
            data_per_year[year][['lat_band', 'dam_count']], 
            on='lat_band', 
            how='left', 
            suffixes=('', f'_{year}')
        )
        merged_df.rename(columns={'dam_count': f'dam_count_{year}'}, inplace=True)
    else:
        merged_df[f'dam_count_{year}'] = 0  # Fill with zeros if data missing for a year

# Fill NA values with 0
for year in years:
    merged_df[f'dam_count_{year}'] = merged_df[f'dam_count_{year}'].fillna(0)

# Define output directory
output_dir = 'E:/wyj/dam/v4/image/visualizations'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. STACKED AREA CHART (Original visualization but with improved styling)
plt.figure(figsize=(24, 6))
plt.stackplot(merged_df['lat_band'], 
              [merged_df[f'dam_count_{year}'] for year in years],
              labels=[str(year) for year in years], 
              colors=colors, 
              alpha=0.8)

# Add legend, title and labels
plt.legend(title='Year', loc='upper right')
plt.xlabel('Latitude (°)', fontsize=14)
plt.ylabel('Number of Dams', fontsize=14)
plt.title('Latitudinal Distribution of Dams - Stacked by Year', fontsize=16)

# Set x-axis ticks
selected_ticks = [-75, -50, -25, 0, 25, 50, 75]
plt.xticks(selected_ticks, fontsize=12)

# Add reference lines
for tick in selected_ticks:
    plt.axvline(tick, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

# Add equator line with special style
plt.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)

# Add grid and set y-axis ticks
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/1_stacked_area_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. MULTI-LINE CHART
plt.figure(figsize=(24, 6))
for i, year in enumerate(years):
    plt.plot(merged_df['lat_band'], 
             merged_df[f'dam_count_{year}'], 
             label=str(year), 
             color=colors[i], 
             linewidth=2.5)

# Add legend, title and labels
plt.legend(title='Year', loc='upper right', fontsize=12)
plt.xlabel('Latitude (°)', fontsize=14)
plt.ylabel('Number of Dams', fontsize=14)
plt.title('Dam Distribution by Latitude - Comparison Across Years', fontsize=16)

# Set axis ticks and grid
plt.xticks(selected_ticks, fontsize=12)
for tick in selected_ticks:
    plt.axvline(tick, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
plt.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)  # Equator
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/2_multiline_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. SMALL MULTIPLES (MULTIPLE PANELS FOR DIFFERENT YEARS)
fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharey=True)
for i, year in enumerate(years):
    axes[i].plot(merged_df['lat_band'], 
                merged_df[f'dam_count_{year}'], 
                color=colors[i], 
                linewidth=2.5)
    axes[i].set_title(f'Dam Distribution in {year}', fontsize=14)
    axes[i].set_xlabel('Latitude (°)', fontsize=12)
    
    # Add vertical reference lines
    for tick in selected_ticks:
        axes[i].axvline(tick, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[i].axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)  # Equator
    
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xticks(selected_ticks)

# Add common y-axis label
fig.text(0.01, 0.5, 'Number of Dams', va='center', rotation='vertical', fontsize=14)
plt.tight_layout()
plt.savefig(f'{output_dir}/3_small_multiples.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. HEATMAP OF LATITUDINAL DISTRIBUTION
# Prepare data for heatmap - filter to reduce visual clutter
filtered_df = merged_df.copy()
filtered_df = filtered_df[(filtered_df['lat_band'] >= -70) & (filtered_df['lat_band'] <= 70)]

# Create pivot table
heatmap_data = filtered_df.set_index('lat_band')[[f'dam_count_{year}' for year in years]].T

# Create heatmap
plt.figure(figsize=(24, 8))
sns.heatmap(heatmap_data, 
            cmap='YlOrRd', 
            robust=True,
            cbar_kws={'label': 'Number of Dams'})

plt.yticks(range(len(years)), years, fontsize=12)
plt.xlabel('Latitude (°)', fontsize=14)
plt.title('Dam Density Heatmap by Latitude Over Time', fontsize=16)
plt.tight_layout()
plt.savefig(f'{output_dir}/4_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. GLOBAL MAP WITH DENSITY REPRESENTATION
fig = plt.figure(figsize=(24, 12))
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()

print('Loading background map...')
# Add map features
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.6)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

print('Loading GeoTIFF file...')

# For the most recent year
year = years[-1]
norm = Normalize(0, merged_df[f'dam_count_{year}'].max())

# Add colored bands based on dam density
for _, row in merged_df.iterrows():
    if row[f'dam_count_{year}'] > 0:
        ax.add_patch(plt.Rectangle(
            (-180, row['lat_band']), 
            360, latitude_band_width, 
            color=plt.cm.YlOrRd(norm(row[f'dam_count_{year}'])),
            transform=ccrs.PlateCarree(),
            alpha=0.7
        ))

print('Adding colorbar...')

# Add colorbar
sm = ScalarMappable(norm=norm, cmap='YlOrRd')
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)

print('Adding gridlines...')
cbar.set_label('Number of Dams', fontsize=14)

# Add gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

plt.title(f'Global Distribution of Dams by Latitude ({year})', fontsize=16)
plt.savefig(f'{output_dir}/5_global_map.png', dpi=300, bbox_inches='tight')
plt.close()

print('The map is complete.')

# 6. PERCENTAGE CHANGE VISUALIZATION
# Calculate percentage change from first to last year
merged_df['change_first_to_last'] = ((merged_df[f'dam_count_{years[-1]}'] - merged_df[f'dam_count_{years[0]}']) / 
                               merged_df[f'dam_count_{years[0]}'] * 100)

# Replace infinity values (from division by zero) with NaN
merged_df['change_first_to_last'] = merged_df['change_first_to_last'].replace([np.inf, -np.inf], np.nan)

# Create percentage change chart
plt.figure(figsize=(24, 6))
mask = ~np.isnan(merged_df['change_first_to_last'])  # Mask to filter out NaN values
bars = plt.bar(merged_df.loc[mask, 'lat_band'], 
              merged_df.loc[mask, 'change_first_to_last'],
              width=latitude_band_width*2,  # Make bars wider for visibility
              color='skyblue')

# Color code bars (positive changes in blue, negative in red)
for i, bar in enumerate(bars):
    if merged_df.loc[mask, 'change_first_to_last'].iloc[i] < 0:
        bar.set_color('#FF9999')  # Light red for negative change
    else:
        bar.set_color('#66B3FF')  # Light blue for positive change

# Add zero reference line
plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)

# Add labels and title
plt.xlabel('Latitude (°)', fontsize=14)
plt.ylabel('Percentage Change (%)', fontsize=14)
plt.title(f'Percentage Change in Dam Count ({years[0]} to {years[-1]})', fontsize=16)

# Set x-axis ticks and add reference lines
plt.xticks(selected_ticks, fontsize=12)
for tick in selected_ticks:
    plt.axvline(tick, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
plt.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)  # Equator

plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/6_percentage_change.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. VIOLIN PLOT BY LATITUDE ZONES
# Define latitude zones
def assign_zone(lat):
    if lat < -66.5:
        return 'Antarctic'
    elif lat < -23.5:
        return 'S. Temperate'
    elif lat < 23.5:
        return 'Tropical'
    elif lat < 66.5:
        return 'N. Temperate'
    else:
        return 'Arctic'

# Create zone data
zone_data = []
for year in years:
    year_data = merged_df[['lat_band', f'dam_count_{year}']].copy()
    year_data['year'] = year
    year_data['zone'] = year_data['lat_band'].apply(assign_zone)
    year_data.rename(columns={f'dam_count_{year}': 'dam_count'}, inplace=True)
    zone_data.append(year_data)

zone_df = pd.concat(zone_data)

# Sum dam counts by zone and year
zone_summary = zone_df.groupby(['zone', 'year'])['dam_count'].sum().reset_index()

# Define custom zone order
zone_order = ['Antarctic', 'S. Temperate', 'Tropical', 'N. Temperate', 'Arctic']

# Create grouped bar chart
plt.figure(figsize=(18, 8))
sns.barplot(x='zone', y='dam_count', hue='year', data=zone_summary, 
           order=zone_order, palette=colors)

plt.title('Dam Distribution by Latitude Zone', fontsize=16)
plt.xlabel('Latitude Zone', fontsize=14)
plt.ylabel('Number of Dams', fontsize=14)
plt.legend(title='Year')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/7_zone_barplot.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. RADIAL PLOT FOR GLOBAL REPRESENTATION
# Prepare data - aggregate to larger latitude bands for clearer visualization
large_band_width = 5  # 5-degree bands for clearer visualization
bands = np.arange(-90, 90 + large_band_width, large_band_width)
labels = bands[:-1]

# Function to aggregate data into larger bands
def aggregate_to_large_bands(df, band_width):
    result = []
    for i in range(len(bands) - 1):
        lower = bands[i]
        upper = bands[i+1]
        band_data = {}
        band_data['lat_band'] = lower
        
        for year in years:
            band_sum = df[(df['lat_band'] >= lower) & (df['lat_band'] < upper)][f'dam_count_{year}'].sum()
            band_data[f'dam_count_{year}'] = band_sum
            
        result.append(band_data)
    
    return pd.DataFrame(result)

# Aggregate data
agg_df = aggregate_to_large_bands(merged_df, large_band_width)

# Convert to radial coordinates - map latitude [-90, 90] to radians [0, 2π]
theta = np.linspace(0, 2*np.pi, len(agg_df))
width = 2*np.pi / len(agg_df)

# Create radial plot
fig = plt.figure(figsize=(14, 14), facecolor='white')
ax = fig.add_subplot(111, polar=True)

for i, year in enumerate(years):
    ax.bar(theta, agg_df[f'dam_count_{year}'], 
           width=width, bottom=0.0, alpha=0.7,
           label=str(year), color=colors[i])

# Configure radial plot appearance
ax.set_theta_zero_location('N')  # 0° at the top (North)
ax.set_theta_direction(-1)  # Clockwise direction
ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
ax.set_xticklabels(['90°N', '45°N', '0°', '45°S', '90°S', '45°S', '0°', '45°N'])
ax.set_title('Global Dam Distribution by Latitude', fontsize=16)
ax.legend(title='Year')

plt.savefig(f'{output_dir}/8_radial_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"All visualizations saved to {output_dir}")