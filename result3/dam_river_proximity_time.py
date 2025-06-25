import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import nearest_points
from rtree import index
import time
import os
from tqdm import tqdm
import fiona
from shapely.geometry import shape

def find_nearest_rivers_to_dams(dam_shp_path, river_gpkg_path, output_dam_path=None, batch_size=1000, max_search_distance=None):
    """
    Find the nearest river segment for each dam and calculate the distance using spatial indexing for efficiency.
    
    Parameters:
    -----------
    dam_shp_path : str
        Path to the dam points shapefile
    river_gpkg_path : str
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
    dict
        Dictionary containing timing information for each step
    """
    # Dictionary to store timing information
    timings = {
        'total': 0,
        'load_dams': 0,
        'load_rivers': 0,
        'build_index': 0,
        'processing': 0,
        'save': 0,
        'batches': {}
    }
    
    total_start_time = time.time()
    
    # Load dam data with timing
    print("Loading dam data...")
    dam_load_start = time.time()
    records = []
    with fiona.open(dam_shp_path) as src:
        src_crs = src.crs
        for i, record in enumerate(src):
            if i >= 5000:  # 限制为前10000条
                break
            records.append(record)
            # 转换为 GeoDataFrame
    
    # Create GeoDataFrame from records and explicitly set the CRS
    dams = gpd.GeoDataFrame(
        [record['properties'] for record in records],
        geometry=[shape(record['geometry']) for record in records],
        crs=src_crs  # Explicitly set the CRS from the source file
    )
     
    
    dam_load_time = time.time() - dam_load_start
    timings['load_dams'] = dam_load_time
    print(f"Dam data loaded in {dam_load_time:.2f} seconds")
    
    # Load river network with timing
    print("Loading river network...")
    river_load_start = time.time()
    rivers = gpd.read_file(river_gpkg_path)
    river_load_time = time.time() - river_load_start
    timings['load_rivers'] = river_load_time
    print(f"River network loaded in {river_load_time:.2f} seconds")
    
    # Ensure both datasets use the same CRS
    if dams.crs != rivers.crs:
        print(f"Reprojecting dams from {dams.crs} to {rivers.crs}")
        reproject_start = time.time()
        dams = dams.to_crs(rivers.crs)
        reproject_time = time.time() - reproject_start
        timings['reproject'] = reproject_time
        print(f"Reprojection completed in {reproject_time:.2f} seconds")
    
    # Add columns for nearest river GOID and distance if they don't exist
    if 'NEAREST_RIVER_GOID' not in dams.columns:
        dams['NEAREST_RIVER_GOID'] = -1  # Default value
    if 'DISTANCE_TO_RIVER' not in dams.columns:
        dams['DISTANCE_TO_RIVER'] = -1.0  # Default value
    
    # Build spatial index for rivers with timing
    print("Building spatial index for rivers...")
    index_start = time.time()
    idx = index.Index()
    for pos, river in enumerate(rivers.itertuples()):
        idx.insert(pos, river.geometry.bounds)
    index_time = time.time() - index_start
    timings['build_index'] = index_time
    print(f"Spatial index built in {index_time:.2f} seconds")
    
    print(f"Processing {len(dams)} dams in batches of {batch_size}...")
    processing_start = time.time()
    
    # Process dams in batches to manage memory
    for batch_start in tqdm(range(0, len(dams), batch_size)):
        batch_end = min(batch_start + batch_size, len(dams))
        batch_id = f"batch_{batch_start}_{batch_end}"
        batch_start_time = time.time()
        
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
        
        # Record batch timing
        batch_time = time.time() - batch_start_time
        timings['batches'][batch_id] = {
            'start_idx': batch_start,
            'end_idx': batch_end,
            'time': batch_time,
            'dams_processed': batch_end - batch_start
        }
        print(f"Batch {batch_id} processed in {batch_time:.2f} seconds")
        
        # Optional: Save intermediate results
        if output_dam_path and batch_end % (batch_size * 10) == 0:
            interim_save_start = time.time()
            
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
            
            interim_save_time = time.time() - interim_save_start
            timings[f'interim_save_{batch_end}'] = interim_save_time
            print(f"Saved interim results to {interim_path} in {interim_save_time:.2f} seconds")
    
    processing_time = time.time() - processing_start
    timings['processing'] = processing_time
    print(f"Processing complete in {processing_time:.2f} seconds")
    
    # Save the updated dam shapefile if path is provided
    if output_dam_path:
        print(f"Saving updated dam data to {output_dam_path}")
        save_start_time = time.time()
        
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
        save_time = time.time() - save_start_time
        timings['save'] = save_time
        print(f"Final save completed in {save_time:.2f} seconds")
    
    # Calculate and record total time
    total_time = time.time() - total_start_time
    timings['total'] = total_time
    print(f"All operations completed in {total_time:.2f} seconds")
    
    # Create timing summary
    print("\nTiming Summary:")
    for step, duration in timings.items():
        if step != 'batches':
            if isinstance(duration, float):
                percentage = (duration / total_time) * 100
                print(f"  {step}: {duration:.2f} seconds ({percentage:.1f}% of total time)")
    
    # Optional: Save timing information to CSV
    if output_dam_path:
        timing_df = pd.DataFrame([
            {'step': step, 'time_seconds': time_val}
            for step, time_val in timings.items()
            if step != 'batches' and isinstance(time_val, float)
        ])
        
        # Add batch timings as separate rows
        for batch_id, batch_info in timings['batches'].items():
            timing_df = pd.concat([timing_df, pd.DataFrame([{
                'step': f"batch_{batch_info['start_idx']}_{batch_info['end_idx']}",
                'time_seconds': batch_info['time'],
                'dams_processed': batch_info['dams_processed'],
                'dams_per_second': batch_info['dams_processed'] / batch_info['time'] if batch_info['time'] > 0 else 0
            }])], ignore_index=True)
            
        timing_path = f"{os.path.splitext(output_dam_path)[0]}_timing_analysis.csv"
        timing_df.to_csv(timing_path, index=False)
        print(f"Detailed timing information saved to {timing_path}")
    
    return dams, timings

def create_dam_river_mapping_table(dams_with_river_info, rivers_gpkg_path, output_csv_path=None):
    """
    Create a mapping table that links dams to their nearest rivers with additional river properties.
    
    Parameters:
    -----------
    dams_with_river_info : GeoDataFrame
        Dam GeoDataFrame with NEAREST_RIVER_GOID and DISTANCE_TO_RIVER columns
    rivers_gpkg_path : str
        Path to the river network GeoPackage
    output_csv_path : str, optional
        Path to save the mapping table as CSV
        
    Returns:
    --------
    DataFrame
        Mapping table linking dams to their nearest rivers
    dict
        Dictionary containing timing information
    """
    timings = {
        'total': 0,
        'load_rivers': 0,
        'create_mapping': 0,
        'save': 0
    }
    
    total_start = time.time()
    
    # Load river network for mapping with timing
    print("Loading river network for mapping...")
    river_load_start = time.time()
    rivers = gpd.read_file(rivers_gpkg_path)
    river_load_time = time.time() - river_load_start
    timings['load_rivers'] = river_load_time
    print(f"River network loaded in {river_load_time:.2f} seconds")
    
    # Get required columns from rivers
    river_columns = ['GOID', 'DIS_AV_CMS', 'RIV_ORD'] 
    river_info = rivers[river_columns].copy()
    
    # Convert dam GeoDataFrame to regular DataFrame with required columns
    dam_columns = ['NEAREST_RIVER_GOID', 'DISTANCE_TO_RIVER']
    dam_info = dams_with_river_info[dam_columns + [col for col in dams_with_river_info.columns 
                                                 if col not in dam_columns + ['geometry']]].copy()
    
    # Merge dam info with river info with timing
    print("Creating mapping table...")
    mapping_start = time.time()
    mapping_table = pd.merge(
        dam_info,
        river_info,
        left_on='NEAREST_RIVER_GOID',
        right_on='GOID',
        how='left'
    )
    mapping_time = time.time() - mapping_start
    timings['create_mapping'] = mapping_time
    print(f"Mapping table created in {mapping_time:.2f} seconds")
    
    if output_csv_path:
        print(f"Saving mapping table to {output_csv_path}")
        save_start = time.time()
        mapping_table.to_csv(output_csv_path, index=False)
        save_time = time.time() - save_start
        timings['save'] = save_time
        print(f"Mapping table saved in {save_time:.2f} seconds")
    
    # Calculate and store total time
    total_time = time.time() - total_start
    timings['total'] = total_time
    print(f"All mapping operations completed in {total_time:.2f} seconds")
    
    return mapping_table, timings

# Example usage with performance enhancements and timing analysis
if __name__ == "__main__":
    # Replace with your actual file paths
    dam_shp_path = "Free-Flowing-Rivers/CODE/数据/dam_shp/YELLOW_RIVER.shp"
    river_gpkg_path = "Free-Flowing-Rivers/CODE/数据/dof_result/origin_yellow.gpkg"
    output_dam_path = "Free-Flowing-Rivers/CODE/数据/dof_result/dam_with_nearest_rivers_speedup.shp"
    
    # Optional: Set a maximum search distance in the units of your CRS (e.g., meters)
    # This speeds up the process by limiting the search radius
    max_search_distance = 5000  # Adjust based on your data context
    
    # Process dams in batches of 1000
    script_start_time = time.time()
    
    print("="*80)
    print("STARTING DAM-RIVER NEAREST NEIGHBOR ANALYSIS")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    updated_dams, dam_timings = find_nearest_rivers_to_dams(
        dam_shp_path=dam_shp_path,
        river_gpkg_path=river_gpkg_path,
        output_dam_path=output_dam_path,
        batch_size=1000,
        max_search_distance=max_search_distance
    )
    
    print("\n" + "="*80)
    print("CREATING DAM-RIVER MAPPING TABLE")
    print("="*80)
    
    # Create a mapping table with timing
    mapping_csv_path = "Free-Flowing-Rivers/CODE/数据/dof_result/dam_river_mapping_speedup.csv"
    mapping_table, mapping_timings = create_dam_river_mapping_table(
        dams_with_river_info=updated_dams,
        rivers_gpkg_path=river_gpkg_path,
        output_csv_path=mapping_csv_path
    )
    
    # Print summary statistics with timing
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total dams processed: {len(updated_dams)}")
    valid_distances = updated_dams['DISTANCE_TO_RIVER'][updated_dams['DISTANCE_TO_RIVER'] >= 0]
    print(f"Dams successfully matched to rivers: {len(valid_distances)}")
    print(f"Average distance to nearest river: {valid_distances.mean():.2f} map units")
    print(f"Maximum distance to nearest river: {valid_distances.max():.2f} map units")
    print(f"Minimum distance to nearest river: {valid_distances.min():.2f} map units")
    
    script_total_time = time.time() - script_start_time
    
    print("\n" + "="*80)
    print("OVERALL TIMING SUMMARY")
    print("="*80)
    print(f"Total script execution time: {script_total_time:.2f} seconds")
    print(f"Dam-river analysis: {dam_timings['total']:.2f} seconds ({dam_timings['total']/script_total_time*100:.1f}%)")
    print(f"Mapping table creation: {mapping_timings['total']:.2f} seconds ({mapping_timings['total']/script_total_time*100:.1f}%)")
    
    # Save overall timing summary
    if output_dam_path:
        overall_timing_df = pd.DataFrame([
            {'step': 'total_script_execution', 'time_seconds': script_total_time},
            {'step': 'dam_river_analysis_total', 'time_seconds': dam_timings['total']},
            {'step': 'mapping_table_creation_total', 'time_seconds': mapping_timings['total']}
        ])
        for step, time_val in dam_timings.items():
            if step != 'batches' and isinstance(time_val, float):
                overall_timing_df = pd.concat([
                    overall_timing_df, 
                    pd.DataFrame([{'step': f'dam_analysis_{step}', 'time_seconds': time_val}])
                ], ignore_index=True)
        
        for step, time_val in mapping_timings.items():
            overall_timing_df = pd.concat([
                overall_timing_df, 
                pd.DataFrame([{'step': f'mapping_{step}', 'time_seconds': time_val}])
            ], ignore_index=True)
        
        overall_timing_path = f"{os.path.splitext(output_dam_path)[0]}_overall_timing.csv"
        overall_timing_df.to_csv(overall_timing_path, index=False)
        print(f"Overall timing summary saved to {overall_timing_path}")
    
    print("\n" + "="*80)
    print(f"ANALYSIS COMPLETED at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)