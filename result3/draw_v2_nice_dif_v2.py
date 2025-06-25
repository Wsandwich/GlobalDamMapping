import geopandas as gpd
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from shapely.geometry import LineString, MultiLineString
import cartopy.crs as ccrs
import fiona
import cartopy.feature as cfeature
from datetime import datetime
import time
from simplification.cutil import simplify_coords

# Define base directory - modify these to match your file structure
base_dir = r"/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/dof_result"
base_dir_2010 = r"/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/dof_result_2010"

output_path = os.path.join(base_dir, "output", "draw")
os.makedirs(output_path, exist_ok=True)

# File paths - replace with your actual file paths
gpkg_filenames = ['7030047060.gpkg']

gpkg_files_2020 = []
gpkg_files_2010 = []

for file in gpkg_filenames:
    file_2020 = os.path.join(base_dir, file)
    gpkg_files_2020.append(file_2020)
    file_2010 = os.path.join(base_dir_2010, file)
    gpkg_files_2010.append(file_2010)

# --------------- Color scheme definition -------------------
# 1. Base colors (for unchanged segments)
# Blue gradient for DOF=0 segments, based on RIV_ORD (river order)
blue_colors = ['#21356A', '#2271B3', '#9BC8DF']  # Deep blue -> Mid blue -> Light blue
blue_cmap = LinearSegmentedColormap.from_list("blue_cmap", blue_colors, N=100)
blue_norm = Normalize(vmin=1, vmax=7)  # RIV_ORD typically ranges from 1 to 7

# Yellow-red gradient for unchanged segments with DOF>0
# yellow_red_colors = ['#FFFFE0', '#FFA475', '#DB4551', '#8B0100']  # Light yellow -> Orange -> Red -> Dark red
yellow_red_colors = ['#FFFFE0', '#FFA475', '#DB4551']  # Light yellow -> Orange -> Red -> Dark red
yellow_red_cmap = LinearSegmentedColormap.from_list("yellow_red_cmap", yellow_red_colors, N=100)
yellow_red_norm = Normalize(vmin=0, vmax=100)  # dof_all typical range

# 2. Change colors (for segments that changed between 2010-2020)
# Purple gradient for DOF increases (positive change)
purple_colors = ['#E6E6FA', '#D8BFD8', '#9932CC', '#4B0082']  # Lavender -> Thistle -> Dark Orchid -> Indigo
purple_cmap = LinearSegmentedColormap.from_list("purple_cmap", purple_colors, N=100)

# Cyan gradient for DOF decreases (negative change)
cyan_colors = ['#E0F3F8', '#80CBC4', '#26A69A', '#00695C']  # Light cyan -> Mid cyan -> Teal -> Deep teal
cyan_colors = ['#00695C', '#26A69A', '#80CBC4', '#E0F3F8'] 
cyan_cmap = LinearSegmentedColormap.from_list("cyan_cmap", cyan_colors, N=100)

# Change norm - for both increase and decrease
change_norm_pos = Normalize(vmin=0, vmax=100)  # For positive changes (increase)
change_norm_neg = Normalize(vmin=-100, vmax=0)  # For negative changes (decrease)

# Width mapping function based on river order
def width_logic(riv_ord):
    # river_width_dict = {
    #     1: 0.3,  # Major rivers
    #     2: 0.2,
    #     3: 0.18,
    #     4: 0.16, 
    #     5: 0.14,
    #     6: 0.12,
    #     7: 0.10  # Smallest tributaries
    # }
    river_width_dict = {}
    river_width_dict[1] = 0.9
    river_width_dict[2] = 0.8
    river_width_dict[3] = 0.7
    river_width_dict[4] = 0.6
    river_width_dict[5] = 0.3
    river_width_dict[6] = 0.2
    river_width_dict[7] = 0.1
    return river_width_dict.get(riv_ord, 0.05)

# Read 2020 data
def read_gpkg_2020(gpkg_file):
    """Read data from 2020 GPKG file"""
    file_name = os.path.basename(gpkg_file)
    print(f"Processing 2020 file: {file_name}")
    
    try:
        # Check if file exists
        if not os.path.exists(gpkg_file):
            print(f"  - File {file_name} does not exist.")
            return None
            
        # Read DOF layer with all required fields
        start_time = time.time()
        dof_gdf = gpd.read_file(gpkg_file, layer='DOF', usecols=['geometry', 'GOID', 'DOF', 'DIS_AV_CMS', 'dof_all', 'RIV_ORD'])
        elapsed_time = time.time() - start_time
        print(f"  - DOF layer records: {len(dof_gdf)} (took {elapsed_time:.2f} seconds)")
        
        # Process GOID as string for joining
        if 'GOID' in dof_gdf.columns:
            dof_gdf['GOID'] = dof_gdf['GOID'].astype(str)
        else:
            print(f"  - Warning: No GOID column found in {file_name}")
            return None
        
        # Process dof_all field
        if 'dof_all' in dof_gdf.columns:
            dof_gdf['dof_all'] = pd.to_numeric(dof_gdf['dof_all'], errors='coerce').fillna(0)
            
            # Add line width based on river order
            if 'RIV_ORD' in dof_gdf.columns:
                start_time = time.time()

                # Filter out small tributaries
                dof_gdf = dof_gdf[dof_gdf['RIV_ORD'] <= 7]
                dof_gdf['width'] = dof_gdf['RIV_ORD'].apply(width_logic)
                elapsed_time = time.time() - start_time
                print(f"  - Applied width_logic and filtered by river order (took {elapsed_time:.2f} seconds)")
            else:
                dof_gdf['width'] = 0.5  # Default width
            
            # Filter non-MultiLineString geometries
            valid_dof_gdf = dof_gdf[dof_gdf.geometry.apply(lambda x: isinstance(x, MultiLineString))]
            
            # Ensure GeoDataFrame has CRS
            if valid_dof_gdf.crs is None:
                valid_dof_gdf.crs = "EPSG:4326"  # Set to WGS 84 if missing
            
            return valid_dof_gdf
            
        else:
            print(f"  - No dof_all field found in DOF layer of {file_name}")
            return None
            
    except Exception as e:
        print(f"Error processing 2020 file {file_name}: {str(e)}")
        return None

# Read 2010 data
def read_gpkg_2010(gpkg_file):
    """Read data from 2010 GPKG file (only GOID and dof_all)"""
    file_name = os.path.basename(gpkg_file)
    print(f"Processing 2010 file: {file_name}")
    
    try:
        # Check if file exists
        if not os.path.exists(gpkg_file):
            print(f"  - File {file_name} does not exist.")
            return None
            
        # Read DOF layer, only including GOID and dof_all fields
        start_time = time.time()
        dof_gdf = gpd.read_file(gpkg_file, layer='DOF', usecols=['GOID', 'dof_all'])
        elapsed_time = time.time() - start_time
        print(f"  - 2010 DOF layer records: {len(dof_gdf)} (took {elapsed_time:.2f} seconds)")
        
        # Process GOID as string for joining
        if 'GOID' in dof_gdf.columns:
            dof_gdf['GOID'] = dof_gdf['GOID'].astype(str)
        else:
            print(f"  - Warning: No GOID column found in 2010 file {file_name}")
            return None
        
        # Process dof_all field
        if 'dof_all' in dof_gdf.columns:
            dof_gdf['dof_all'] = pd.to_numeric(dof_gdf['dof_all'], errors='coerce').fillna(0)
            return dof_gdf
        else:
            print(f"  - No dof_all field found in 2010 DOF layer of {file_name}")
            return None
            
    except Exception as e:
        print(f"Error processing 2010 file {file_name}: {str(e)}")
        return None

# Calculate differences between 2010 and 2020
def calculate_dof_diff_by_goid(gdf_2020, gdf_2010_data):
    """Calculate dof_all differences between 2020 and 2010 using GOID matching"""
    if gdf_2020 is None:
        print("No 2020 data available for comparison")
        return None
        
    if gdf_2010_data is None:
        print("No 2010 data found, assuming all 2010 dof_all values are 0")
        # Copy 2020 data, use dof_all directly as difference
        diff_gdf = gdf_2020.copy()
        diff_gdf['dof_diff'] = diff_gdf['dof_all']
        diff_gdf['status'] = 'new'  # All segments are considered new/changed
        return diff_gdf
    
    # Create difference GeoDataFrame
    diff_gdf = gdf_2020.copy()
    diff_gdf['dof_diff'] = 0  # Default difference is 0
    diff_gdf['status'] = 'unchanged'  # Default status
    
    # Create mapping from GOID to dof_all for 2010 data
    dof_2010_map = dict(zip(gdf_2010_data['GOID'], gdf_2010_data['dof_all']))
    
    # Calculate differences
    start_time = time.time()
    change_threshold = 0.5  # Minimum change to consider a segment as changed
    
    for idx, row in diff_gdf.iterrows():
        goid = row['GOID']
        if goid in dof_2010_map:
            # Calculate difference if GOID found in 2010 data
            diff = row['dof_all'] - dof_2010_map[goid]
            diff_gdf.loc[idx, 'dof_diff'] = diff
            
            # Determine status based on difference magnitude
            if abs(diff) > change_threshold:
                if diff > 0:
                    diff_gdf.loc[idx, 'status'] = 'increased'
                else:
                    diff_gdf.loc[idx, 'status'] = 'decreased'
            else:
                diff_gdf.loc[idx, 'status'] = 'unchanged'
        else:
            # If GOID not found in 2010, difference is the 2020 value (new segment)
            diff_gdf.loc[idx, 'dof_diff'] = row['dof_all']
            diff_gdf.loc[idx, 'status'] = 'new'
    
    elapsed_time = time.time() - start_time
    print(f"  - Calculated dof_diff for {len(diff_gdf)} records (took {elapsed_time:.2f} seconds)")
    
    # Print summary statistics
    status_counts = diff_gdf['status'].value_counts()
    print(f"  - Status distribution: {status_counts.to_dict()}")
    print(f"  - dof_diff range: min={diff_gdf['dof_diff'].min()}, max={diff_gdf['dof_diff'].max()}")
    
    return diff_gdf

# Plot combined status and change map
def plot_combined_map(output_file, diff_gdf, title="River Fragmentation Status and Change (2010-2020)", 
                     hatch_pattern='////////////', hatch_color='#DDD4DA', hatch_linewidth=0.2):
    """Plot combined river fragmentation status and change map"""
    if diff_gdf is None or len(diff_gdf) == 0:
        print("No data to plot")
        return
    
    print("Starting combined visualization...")
    
    # Set global hatching properties for land areas
    plt.rcParams['hatch.color'] = hatch_color
    plt.rcParams['hatch.linewidth'] = hatch_linewidth

    # Create figure with Robinson projection
    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.Robinson()})
    
    # Add land base map with hatching
    try:
        # Try to load world map for context
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        has_world_map = True
        print("Successfully loaded world map")
        
        land = cfeature.NaturalEarthFeature(
            'physical', 'land', '10m',
            facecolor='#f0f0f0',  # Light gray
            hatch=hatch_pattern  # Diagonal lines
        )
        ax.add_feature(land)
    except Exception as e:
        print(f"Error loading world map: {e}")
        # Fallback to simple land feature
        ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)
    
    # Organize data by status category for separate rendering
    status_categories = {
        'unchanged_zero': diff_gdf[(diff_gdf['status'] == 'unchanged') & (diff_gdf['dof_all'] == 0)],
        'unchanged_nonzero': diff_gdf[(diff_gdf['status'] == 'unchanged') & (diff_gdf['dof_all'] > 0)],
        'increased': diff_gdf[diff_gdf['status'] == 'increased'],
        'decreased': diff_gdf[diff_gdf['status'] == 'decreased'],
        'new': diff_gdf[diff_gdf['status'] == 'new']
    }
    
    # Function to process geometry for LineCollection
    def process_geometry_for_lines(geom_series, attribute_series, width_series):
        segments = []
        values = []
        widths = []
        
        for geom, attr, width in zip(geom_series, attribute_series, width_series):
            if isinstance(geom, LineString):
                segments.append(list(geom.coords))
                values.append(attr)
                widths.append(width)
            elif isinstance(geom, MultiLineString):
                for line in geom.geoms:
                    segments.append(list(line.coords))
                    values.append(attr)
                    widths.append(width)
        
        return segments, values, widths
    
    print("Processing geometries...")
    
    # Process each category separately for better rendering control
    for status, data in status_categories.items():
        if data.empty:
            continue
        
        print(f"Processing {status} segments: {len(data)} features")
        
        # Use projected geometry if available, otherwise use original geometry
        geom_col = 'projected_geom' if 'projected_geom' in data.columns else 'geometry'
        
        if status == 'unchanged_zero':
            # Blue gradient for unchanged DOF=0 segments
            segments, values, widths = process_geometry_for_lines(
                data[geom_col], data['RIV_ORD'], data['width']
            )

            simplified_segments = [simplify_coords(seg, epsilon=0.5) for seg in segments]

            if simplified_segments:
                lc = LineCollection(
                    simplified_segments,
                    linewidths=widths,
                    cmap=blue_cmap,
                    norm=blue_norm,
                    transform=ccrs.PlateCarree(),
                    zorder=10,
                    capstyle='round',
                    joinstyle='round'
                )
                lc.set_array(np.array(values))
                ax.add_collection(lc)
                
        elif status == 'unchanged_nonzero':
            # Yellow-red gradient for unchanged DOF>0 segments
            segments, values, widths = process_geometry_for_lines(
                data[geom_col], data['dof_all'], data['width']
            )

            simplified_segments = [simplify_coords(seg, epsilon=0.5) for seg in segments]
            
            if simplified_segments:
                lc = LineCollection(
                    simplified_segments,
                    linewidths=widths,
                    cmap=yellow_red_cmap,
                    norm=yellow_red_norm,
                    transform=ccrs.PlateCarree(),
                    zorder=11,
                    capstyle='round',
                    joinstyle='round',
                    alpha=0.5  # 设置透明度，例如 0.5 表示半透明
                )
                lc.set_array(np.array(values))
                ax.add_collection(lc)
                
        elif status in ['increased', 'new']:
            # Orange gradient for increased or new DOF
            segments, values, widths = process_geometry_for_lines(
                data[geom_col], data['dof_diff'], data['width']
            )

            simplified_segments = [simplify_coords(seg, epsilon=0.5) for seg in segments]
            
            if simplified_segments:
                # Filter out negative values (shouldn't happen in this category)
                pos_segments = []
                pos_values = []
                pos_widths = []
                
                for seg, val, wid in zip(simplified_segments, values, widths):
                    if val >= 0:
                        pos_segments.append(seg)
                        pos_values.append(val)
                        pos_widths.append(wid)
                
                if pos_segments:
                    lc = LineCollection(
                        pos_segments,
                        linewidths=pos_widths,
                        cmap=purple_cmap,
                        norm=change_norm_pos,
                        transform=ccrs.PlateCarree(),
                        zorder=13,  # Higher zorder to be on top
                        capstyle='round',
                        joinstyle='round'
                    )
                    lc.set_array(np.array(pos_values))
                    ax.add_collection(lc)
                
        elif status == 'decreased':
            # Cyan gradient for decreased DOF
            segments, values, widths = process_geometry_for_lines(
                data[geom_col], data['dof_diff'], data['width']
            )
            
            simplified_segments = [simplify_coords(seg, epsilon=0.5) for seg in segments]

            if simplified_segments:
                # Negate values for proper color mapping (negative values)
                neg_segments = []
                neg_values = []
                neg_widths = []
                
                for seg, val, wid in zip(simplified_segments, values, widths):
                    if val < 0:
                        neg_segments.append(seg)
                        neg_values.append(val)  # Keep negative
                        neg_widths.append(wid)
                
                if neg_segments:
                    lc = LineCollection(
                        neg_segments,
                        linewidths=neg_widths,
                        cmap=cyan_cmap,
                        norm=change_norm_neg,
                        transform=ccrs.PlateCarree(),
                        zorder=12,  # Higher zorder to be on top
                        capstyle='round',
                        joinstyle='round'
                    )
                    lc.set_array(np.array(neg_values))
                    ax.add_collection(lc)
    
    # Set global map view
    ax.set_global()
    
    # Set title
    ax.set_title(title, fontsize=16, pad=20)
    
    # Create legend with color bars
    # 1. Blue gradient for unchanged DOF=0 by river order
    sm_blue = ScalarMappable(cmap=blue_cmap, norm=blue_norm)
    sm_blue.set_array([])
    cbar_blue = plt.colorbar(sm_blue, ax=ax, orientation='horizontal', fraction=0.023, pad=0.05, 
                            location='bottom', shrink=0.6)
    cbar_blue.set_label('River Order (DOF=0, Unchanged)', fontsize=10)
    
    # 2. Yellow-red gradient for unchanged DOF>0
    sm_yellow_red = ScalarMappable(cmap=yellow_red_cmap, norm=yellow_red_norm)
    sm_yellow_red.set_array([])
    cbar_yellow_red = plt.colorbar(sm_yellow_red, ax=ax, orientation='horizontal', fraction=0.023, pad=0.10, 
                                   location='bottom', shrink=0.6)
    cbar_yellow_red.set_label('DOF Score (>0, Unchanged)', fontsize=10)
    
    # 3. Purple gradient for increased DOF
    sm_orange = ScalarMappable(cmap=purple_cmap, norm=change_norm_pos)
    sm_orange.set_array([])
    cbar_orange = plt.colorbar(sm_orange, ax=ax, orientation='horizontal', fraction=0.023, pad=0.15, 
                              location='bottom', shrink=0.6)
    cbar_orange.set_label('DOF Increase (2010-2020)', fontsize=10)
    
    # 4. Cyan gradient for decreased DOF
    sm_cyan = ScalarMappable(cmap=cyan_cmap, norm=change_norm_neg)
    sm_cyan.set_array([])
    cbar_cyan = plt.colorbar(sm_cyan, ax=ax, orientation='horizontal', fraction=0.023, pad=0.20, 
                            location='bottom', shrink=0.6)
    cbar_cyan.set_label('DOF Decrease (2010-2020)', fontsize=10)
    
    # Add metadata text
    plt.figtext(0.5, 0.02, "River Fragmentation Analysis (2010-2020) • Dataset: 4.1 million dams", 
               ha='center', fontsize=10, style='italic')
    
    # Save high-resolution image
    start_time = time.time()
    plt.savefig(output_file, dpi=2000, bbox_inches='tight')
    elapsed_time = time.time() - start_time
    print(f"Map saved to {output_file} (took {elapsed_time:.2f} seconds)")
    plt.close(fig)

# Main function
def main():
    # Get current time for filename
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("Starting 2010-2020 comparison and visualization...")
    all_diff_data = []
    
    # Process each file pair and calculate differences
    for i, gpkg_file_2020 in enumerate(gpkg_files_2020):
        # Get filename
        file_name = os.path.basename(gpkg_file_2020)
        
        # Read 2020 data (complete spatial and attribute information)
        gdf_2020 = read_gpkg_2020(gpkg_file_2020)
        if gdf_2020 is None:
            print(f"Unable to read 2020 file: {file_name}, skipping comparison")
            continue
        
        # Check for corresponding 2010 file
        gpkg_file_2010 = None
        if i < len(gpkg_files_2010):
            gpkg_file_2010 = gpkg_files_2010[i]
        
        # Read 2010 data (if exists, only GOID and dof_all)
        gdf_2010_data = None
        if gpkg_file_2010 and os.path.exists(gpkg_file_2010):
            print(f"Reading corresponding 2010 data: {os.path.basename(gpkg_file_2010)}")
            gdf_2010_data = read_gpkg_2010(gpkg_file_2010)
        else:
            print(f"No corresponding 2010 data file found, assuming all 2010 dof_all values are 0")
        
        # Calculate differences based on GOID
        diff_gdf = calculate_dof_diff_by_goid(gdf_2020, gdf_2010_data)
        
        if diff_gdf is not None and not diff_gdf.empty:
            all_diff_data.append(diff_gdf)
            
            # Create visualization for individual file
            output_file = os.path.join(output_path, f"combined_status_change_{file_name.replace('.gpkg', '')}_{current_time}.png")
            plot_combined_map(output_file, diff_gdf, f"River Fragmentation Status & Change (2010-2020) - {file_name}")
    
    # If multiple files, create combined visualization
    if len(all_diff_data) > 1:
        print(f"Combining {len(all_diff_data)} datasets for global visualization...")
        combined_output_file = os.path.join(output_path, f"global_status_change_{current_time}.png")
        # Combine all difference data using GeoDataFrame's concat
        combined_diff_gdf = pd.concat(all_diff_data)
        plot_combined_map(combined_output_file, combined_diff_gdf, "Global River Fragmentation Status & Change (2010-2020)")
    
    print("Analysis and visualization complete!")

if __name__ == "__main__":
    main()