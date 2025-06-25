'''
This code uses 1h50m
'''

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import nearest_points
from rtree import index
import time
import os
from tqdm import tqdm

def find_nearest_rivers_to_dams(dam_shp_path, rivers, output_dam_path=None, batch_size=1000, max_search_distance=None):
    """
    Find the nearest river segment for each dam and calculate the distance using spatial indexing for efficiency.
    
    Parameters:
    -----------
    dam_shp_path : str
        Path to the dam points shapefile
    rivers : file
        Path to the river network GeoPackage
    output_dam_path : str, optional
        Path to save the updated dam shapefile. If None, returns the GeoDataFrame
    batch_size : int, optional
        Number of dams to process in each batch to manage memory usage
    max_search_distance : float, optional
        Maximum distance to search for nearest river. If provided, speeds up processing
        by limiting the search radius. Units should match the CRS units.
        
    Returns:
    --------
    GeoDataFrame
        Updated dam GeoDataFrame with nearest river GOID and distance
    """
    start_time = time.time()
    print("Loading dam data...")
    dams = gpd.read_file(dam_shp_path)
    
    print("Loading river network...")

    # Ensure both datasets use the same CRS
    if dams.crs != rivers.crs:
        print(f"Reprojecting dams from {dams.crs} to {rivers.crs}")
        dams = dams.to_crs(rivers.crs)
    
    # Add columns for nearest river GOID and distance if they don't exist
    if 'NEAREST_RIVER_GOID' not in dams.columns:
        dams['NEAREST_RIVER_GOID'] = -1  # Default value
    if 'DISTANCE_TO_RIVER' not in dams.columns:
        dams['DISTANCE_TO_RIVER'] = -1.0  # Default value
    
    # Build spatial index for rivers
    print("Building spatial index for rivers...")
    idx = index.Index()
    for pos, river in enumerate(rivers.itertuples()):
        idx.insert(pos, river.geometry.bounds)
    
    print(f"Processing {len(dams)} dams in batches of {batch_size}...")
    
    # Process dams in batches to manage memory
    for batch_start in tqdm(range(0, len(dams), batch_size)):
        batch_end = min(batch_start + batch_size, len(dams))
        dam_batch = dams.iloc[batch_start:batch_end].copy()
        
        # Process each dam in the batch
        for i, dam in enumerate(dam_batch.itertuples()):
            dam_idx = batch_start + i
            dam_geom = dam.geometry
            
            # Get potential nearest rivers using spatial index
            if max_search_distance:
                # If we have a max search distance, create a bounding box around the dam
                minx, miny, maxx, maxy = (
                    dam_geom.x - max_search_distance,
                    dam_geom.y - max_search_distance,
                    dam_geom.x + max_search_distance,
                    dam_geom.y + max_search_distance
                )
                potential_matches_idx = list(idx.intersection((minx, miny, maxx, maxy)))
            else:
                # Otherwise use the nearest 50 rivers by bounding box for efficiency
                dam_bounds = dam_geom.buffer(0.1).bounds  # Small buffer to ensure we get some matches
                potential_matches_idx = list(idx.nearest(dam_bounds, 50))
            
            if not potential_matches_idx:
                continue  # Skip if no potential matches found
                
            # Get the river geometries for the potential matches
            potential_rivers = rivers.iloc[potential_matches_idx]
            
            # Calculate distances to potential river segments
            distances = potential_rivers.geometry.distance(dam_geom)
            
            # Find the index of the nearest river segment
            if len(distances) > 0:
                min_local_idx = distances.argmin()
                min_distance = distances.iloc[min_local_idx]
                
                # Get the GOID of the nearest river segment
                nearest_river_goid = potential_rivers.iloc[min_local_idx]['GOID']
                
                # Update dam properties in the main dataframe
                dams.at[dam_idx, 'NEAREST_RIVER_GOID'] = nearest_river_goid
                dams.at[dam_idx, 'DISTANCE_TO_RIVER'] = min_distance
        
        # Optional: Save intermediate results
        if output_dam_path and batch_end % (batch_size * 10) == 0:
            # Fix field names with non-Latin characters before saving to shapefile
            dams_for_save = dams.copy()
            
            # Map of problematic field names to safe alternatives
            renamed_fields = {}
            for col in dams_for_save.columns:
                # Check if column name contains non-ASCII characters
                if not all(ord(c) < 128 for c in col):
                    # Create a safe name: 'field_X' where X is a number
                    safe_name = f"field_{len(renamed_fields) + 1}"
                    renamed_fields[col] = safe_name
                    # Rename the column
                    dams_for_save = dams_for_save.rename(columns={col: safe_name})
            
            # Log the renamed fields for reference
            if renamed_fields:
                print("Renamed the following fields to save to shapefile:")
                for orig, new_name in renamed_fields.items():
                    print(f"  '{orig}' → '{new_name}'")
            
            interim_path = f"{os.path.splitext(output_dam_path)[0]}_interim.shp"
            dams_for_save.to_file(interim_path)
            print(f"Saved interim results to {interim_path}")
    
    print(f"Processing complete. Total time: {time.time() - start_time:.2f} seconds")
    
    # Save the updated dam shapefile if path is provided
    if output_dam_path:
        print(f"Saving updated dam data to {output_dam_path}")
        
        # Fix field names with non-Latin characters before saving to shapefile
        dams_for_save = dams.copy()
        
        # Map of problematic field names to safe alternatives
        renamed_fields = {}
        for col in dams_for_save.columns:
            # Check if column name contains non-ASCII characters
            if not all(ord(c) < 128 for c in col):
                # Create a safe name: 'field_X' where X is a number
                safe_name = f"field_{len(renamed_fields) + 1}"
                renamed_fields[col] = safe_name
                # Rename the column
                dams_for_save = dams_for_save.rename(columns={col: safe_name})
        
        # Log the renamed fields for reference
        if renamed_fields:
            print("Renamed the following fields to save to shapefile:")
            for orig, new_name in renamed_fields.items():
                print(f"  '{orig}' → '{new_name}'")
            
            # Create a mapping file to help remember the original field names
            mapping_df = pd.DataFrame(list(renamed_fields.items()), columns=['Original_Name', 'Shapefile_Name'])
            mapping_path = f"{os.path.splitext(output_dam_path)[0]}_field_mapping.csv"
            mapping_df.to_csv(mapping_path, index=False, encoding='utf-8')
            print(f"Field name mapping saved to {mapping_path}")
        
        dams_for_save.to_file(output_dam_path)
    
    return dams

def create_dam_river_mapping_table(dams_with_river_info, rivers, output_csv_path=None):
    """
    Create a mapping table that links dams to their nearest rivers with additional river properties.
    
    Parameters:
    -----------
    dams_with_river_info : GeoDataFrame
        Dam GeoDataFrame with NEAREST_RIVER_GOID and DISTANCE_TO_RIVER columns
    rivers : str
        Path to the river network GeoPackage
    output_csv_path : str, optional
        Path to save the mapping table as CSV
        
    Returns:
    --------
    DataFrame
        Mapping table linking dams to their nearest rivers
    """
    print("Loading river network for mapping...")

    # Get required columns from rivers
    river_columns = ['GOID', 'DIS_AV_CMS', 'RIV_ORD'] 
    river_info = rivers[river_columns].copy()
    
    # Convert dam GeoDataFrame to regular DataFrame with required columns
    dam_columns = ['NEAREST_RIVER_GOID', 'DISTANCE_TO_RIVER']
    dam_info = dams_with_river_info[dam_columns + [col for col in dams_with_river_info.columns 
                                                 if col not in dam_columns + ['geometry']]].copy()
    
    # Merge dam info with river info
    print("Creating mapping table...")
    mapping_table = pd.merge(
        dam_info,
        river_info,
        left_on='NEAREST_RIVER_GOID',
        right_on='GOID',
        how='left'
    )
    
    if output_csv_path:
        print(f"Saving mapping table to {output_csv_path}")
        mapping_table.to_csv(output_csv_path, index=False)
    
    return mapping_table

# Example usage with performance enhancements
if __name__ == "__main__":
    # Replace with your actual file paths
    dam_shp_path = "Free-Flowing-Rivers/CODE/数据/dam_shp/YELLOW_RIVER.shp"
    river_gpkg_path = "Free-Flowing-Rivers/CODE/数据/dof_result/origin_yellow.gpkg"
    output_dam_path = "Free-Flowing-Rivers/CODE/数据/dof_result/dam_with_nearest_rivers_speedup.shp"
    
    # Optional: Set a maximum search distance in the units of your CRS (e.g., meters)
    # This speeds up the process by limiting the search radius
    max_search_distance = 5000  # Adjust based on your data context
    
    rivers = gpd.read_file(river_gpkg_path, columns=['geometry', 'GOID', 'DIS_AV_CMS', 'RIV_ORD'])
    
    # Process dams in batches of 1000
    updated_dams = find_nearest_rivers_to_dams(
        dam_shp_path=dam_shp_path,
        rivers=rivers,
        output_dam_path=output_dam_path,
        batch_size=1000,
        max_search_distance=max_search_distance
    )
    
    # Create a mapping table (optional)
    mapping_csv_path = "Free-Flowing-Rivers/CODE/数据/dof_result/dam_river_mapping_speedup.csv"
    mapping_table = create_dam_river_mapping_table(
        dams_with_river_info=updated_dams,
        rivers=rivers,
        output_csv_path=mapping_csv_path
    )
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total dams processed: {len(updated_dams)}")
    valid_distances = updated_dams['DISTANCE_TO_RIVER'][updated_dams['DISTANCE_TO_RIVER'] >= 0]
    print(f"Dams successfully matched to rivers: {len(valid_distances)}")
    print(f"Average distance to nearest river: {valid_distances.mean():.2f} map units")
    print(f"Maximum distance to nearest river: {valid_distances.max():.2f} map units")
    print(f"Minimum distance to nearest river: {valid_distances.min():.2f} map units")