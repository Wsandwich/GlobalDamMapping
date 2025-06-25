import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import nearest_points
from rtree import index
import time
import os
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Define the CRS to use for projection - keeping original CRS to avoid matching issues
TARGET_CRS = None  # We'll keep the original CRS instead of forcing a projection

def find_nearest_rivers_to_dams(dam_shp_path, rivers=None, river_gpkg_path=None, output_dam_path=None, 
                                batch_size=1000, max_search_distance=None, n_jobs=-1):
    """
    Find the nearest river segment for each dam and calculate the distance using spatial indexing for efficiency.
    Optimized version with parallelization using ProcessPoolExecutor and performance improvements.
    
    Parameters:
    -----------
    dam_shp_path : str
        Path to the dam points shapefile
    rivers : GeoDataFrame, optional
        Pre-loaded river network data to avoid reloading
    river_gpkg_path : str, optional
        Path to the river network GeoPackage (used only if rivers is None)
    output_dam_path : str, optional
        Path to save the updated dam shapefile. If None, returns the GeoDataFrame
    batch_size : int, optional
        Number of dams to process in each batch for parallel processing
    max_search_distance : float, optional
        Maximum distance to search for nearest river in map units
    n_jobs : int, optional
        Number of parallel jobs to run. Default -1 uses all available cores except one.
        
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
        'reproject': 0,
        'build_index': 0,
        'processing': 0,
        'save': 0,
        'batches': {}
    }
    
    total_start_time = time.time()
    
    # Load dam data with timing
    print("Loading dam data...")
    dam_load_start = time.time()
    dams = gpd.read_file(dam_shp_path)
    dam_load_time = time.time() - dam_load_start
    timings['load_dams'] = dam_load_time
    print(f"Dam data loaded in {dam_load_time:.2f} seconds")
    
    # Load river network if not provided
    if rivers is None:
        print("Loading river network...")
        river_load_start = time.time()
        # Load only necessary columns if available
        try:
            # Try loading only essential columns
            rivers = gpd.read_file(river_gpkg_path, columns=['geometry', 'GOID', 'DIS_AV_CMS', 'RIV_ORD'])
        except:
            # If specific columns aren't supported, load everything
            rivers = gpd.read_file(river_gpkg_path)
        river_load_time = time.time() - river_load_start
        timings['load_rivers'] = river_load_time
        print(f"River network loaded in {river_load_time:.2f} seconds")
    else:
        print("Using pre-loaded river network")
    
    # Ensure both datasets use the same CRS
    reproject_start = time.time()
    
    # Check if CRS is different and reproject if necessary
    if dams.crs != rivers.crs:
        print(f"Reprojecting dams from {dams.crs} to {rivers.crs}")
        dams = dams.to_crs(rivers.crs)
    
    reproject_time = time.time() - reproject_start
    timings['reproject'] = reproject_time
    print(f"CRS check/reprojection completed in {reproject_time:.2f} seconds")
    
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
    
    # Function to process a single dam
    def process_dam(dam_idx, dam, max_search_distance):
        dam_geom = dam.geometry
        
        # Get potential nearest rivers using spatial index with adaptive buffer
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
            return dam_idx, -1, -1.0  # No matches found
        
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
            
            return dam_idx, nearest_river_goid, min_distance
        
        return dam_idx, -1, -1.0  # Fallback return
    
    # Function to process a batch of dams
    def process_batch(batch_info):
        start_idx, end_idx = batch_info
        batch_start_time = time.time()
        batch_results = []
        
        # Process each dam in the batch
        dams_batch = dams.iloc[start_idx:end_idx]
        for i, dam in enumerate(dams_batch.itertuples()):
            dam_idx = start_idx + i
            result = process_dam(dam_idx, dam, max_search_distance)
            batch_results.append(result)
        
        batch_time = time.time() - batch_start_time
        return batch_results, batch_time, (start_idx, end_idx)
    
    # Determine optimal number of jobs
    if n_jobs == -1:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)  # Use N-1 cores
    
    print(f"Processing {len(dams)} dams in batches of {batch_size} using {n_jobs} parallel jobs...")
    processing_start = time.time()
    
    # Prepare batches
    batch_indices = [(i, min(i + batch_size, len(dams))) for i in range(0, len(dams), batch_size)]
    
    # Process batches in parallel using ProcessPoolExecutor
    all_results = []
    all_batch_times = {}
    
    # Create a progress bar
    progress_bar = tqdm(total=len(batch_indices))
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all batch jobs
        future_to_batch = {executor.submit(process_batch, batch_info): batch_info for batch_info in batch_indices}
        
        # Process results as they complete
        for future in as_completed(future_to_batch):
            batch_info = future_to_batch[future]
            try:
                batch_results, batch_time, (start_idx, end_idx) = future.result()
                all_results.extend(batch_results)
                
                # Record timing information
                batch_id = f"batch_{start_idx}_{end_idx}"
                all_batch_times[batch_id] = {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'time': batch_time,
                    'dams_processed': end_idx - start_idx
                }
                
                # Log batch completion
                print(f"Batch {batch_id} processed in {batch_time:.2f} seconds")
                
                # Update progress bar
                progress_bar.update(1)
                
            except Exception as exc:
                print(f"Batch {batch_info} generated an exception: {exc}")
    
    # Close progress bar
    progress_bar.close()
    
    # Update timings with batch information
    timings['batches'] = all_batch_times
    
    # Update dam dataframe with results
    for dam_idx, goid, distance in all_results:
        dams.at[dam_idx, 'NEAREST_RIVER_GOID'] = goid
        dams.at[dam_idx, 'DISTANCE_TO_RIVER'] = distance
    
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
                'step': batch_id,
                'time_seconds': batch_info['time'],
                'dams_processed': batch_info['dams_processed'],
                'dams_per_second': batch_info['dams_processed'] / batch_info['time'] if batch_info['time'] > 0 else 0
            }])], ignore_index=True)
            
        timing_path = f"{os.path.splitext(output_dam_path)[0]}_timing_analysis.csv"
        timing_df.to_csv(timing_path, index=False)
        print(f"Detailed timing information saved to {timing_path}")
    
    return dams, timings


def create_dam_river_mapping_table(dams_with_river_info, rivers=None, rivers_gpkg_path=None, output_csv_path=None):
    """
    Create a mapping table that links dams to their nearest rivers with additional river properties.
    Optimized to use pre-loaded river data when available.
    
    Parameters:
    -----------
    dams_with_river_info : GeoDataFrame
        Dam GeoDataFrame with NEAREST_RIVER_GOID and DISTANCE_TO_RIVER columns
    rivers : GeoDataFrame, optional
        Pre-loaded river network data to avoid reloading
    rivers_gpkg_path : str, optional
        Path to the river network GeoPackage (used only if rivers is None)
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
    
    # Load river network for mapping if not provided
    if rivers is None and rivers_gpkg_path is not None:
        print("Loading river network for mapping...")
        river_load_start = time.time()
        try:
            # Try loading only essential columns
            rivers = gpd.read_file(rivers_gpkg_path, columns=['GOID', 'DIS_AV_CMS', 'RIV_ORD'])
        except:
            # If specific columns aren't supported, load everything
            rivers = gpd.read_file(rivers_gpkg_path)
        river_load_time = time.time() - river_load_start
        timings['load_rivers'] = river_load_time
        print(f"River network loaded in {river_load_time:.2f} seconds")
    else:
        print("Using pre-loaded river network")
    
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

# Example usage with optimizations
if __name__ == "__main__":
    # Replace with your actual file paths
    dam_shp_path = "Free-Flowing-Rivers/CODE/数据/dam_shp/YELLOW_RIVER.shp"
    river_gpkg_path = "Free-Flowing-Rivers/CODE/数据/dof_result/origin_yellow.gpkg"
    output_dam_path = "Free-Flowing-Rivers/CODE/数据/dof_result/dam_with_nearest_rivers_optimized_v2.shp"
    
    # Optional: Set a maximum search distance in map units
    # For geographic coordinates, a reasonable value might be ~0.1 degrees
    # Adjust based on your data - we'll use 5000 units as a starting point
    max_search_distance = 5000
    
    # Settings for parallel processing
    batch_size = 500  # Smaller batches for better parallel distribution
    n_jobs = -1  # Use all available cores minus one
    
    script_start_time = time.time()
    
    print("="*80)
    print("STARTING DAM-RIVER NEAREST NEIGHBOR ANALYSIS")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Load rivers once to avoid loading twice
    print("Loading river network (shared loading)...")
    river_load_start = time.time()
    try:
        # Try loading only essential columns
        rivers = gpd.read_file(river_gpkg_path, columns=['geometry', 'GOID', 'DIS_AV_CMS', 'RIV_ORD'])
    except:
        # If specific columns aren't supported, load everything
        rivers = gpd.read_file(river_gpkg_path)
    river_load_time = time.time() - river_load_start
    print(f"River network loaded in {river_load_time:.2f} seconds")
    
    # Process dams using the pre-loaded river network
    updated_dams, dam_timings = find_nearest_rivers_to_dams(
        dam_shp_path=dam_shp_path,
        rivers=rivers,  # Pass pre-loaded rivers
        output_dam_path=output_dam_path,
        batch_size=batch_size,
        max_search_distance=max_search_distance,
        n_jobs=n_jobs
    )
    
    print("\n" + "="*80)
    print("CREATING DAM-RIVER MAPPING TABLE")
    print("="*80)
    
    # Create a mapping table with timing using the same pre-loaded river data
    mapping_csv_path = "Free-Flowing-Rivers/CODE/数据/dof_result/dam_river_mapping_optimized.csv"
    mapping_table, mapping_timings = create_dam_river_mapping_table(
        dams_with_river_info=updated_dams,
        rivers=rivers,  # Pass pre-loaded rivers
        output_csv_path=mapping_csv_path
    )
    
    # Print summary statistics with timing
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total dams processed: {len(updated_dams)}")
    valid_distances = updated_dams['DISTANCE_TO_RIVER'][updated_dams['DISTANCE_TO_RIVER'] >= 0]
    if len(valid_distances) > 0:
        print(f"Dams successfully matched to rivers: {len(valid_distances)}")
        print(f"Average distance to nearest river: {valid_distances.mean():.6f} map units")
        print(f"Maximum distance to nearest river: {valid_distances.max():.6f} map units")
        print(f"Minimum distance to nearest river: {valid_distances.min():.6f} map units")
    else:
        print("No dams were successfully matched to rivers. Check your data and search distance.")
    
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
    
    # Calculate serial time for comparison
    serial_time_estimate = sum([batch_info['time'] for batch_id, batch_info in dam_timings['batches'].items()])
    if dam_timings['processing'] > 0:
        parallelization_speedup = serial_time_estimate / dam_timings['processing'] 
        print("\n" + "="*80)
        print("PERFORMANCE IMPROVEMENT ESTIMATE")
        print("="*80)
        print(f"Estimated parallelization speedup: {parallelization_speedup:.2f}x")
        print(f"Estimated time saved from parallelization: {serial_time_estimate - dam_timings['processing']:.2f} seconds")
    
        # Save performance improvement metrics
        if output_dam_path:
            improvement_df = pd.DataFrame([
                {'metric': 'total_execution_time', 'value': script_total_time},
                {'metric': 'parallelization_speedup', 'value': parallelization_speedup},
                {'metric': 'time_saved_from_parallelization', 'value': serial_time_estimate - dam_timings['processing']},
                {'metric': 'dams_per_second', 'value': len(updated_dams) / dam_timings['processing'] if dam_timings['processing'] > 0 else 0}
            ])
            improvement_path = f"{os.path.splitext(output_dam_path)[0]}_performance_metrics.csv"
            improvement_df.to_csv(improvement_path, index=False)
            print(f"Performance metrics saved to {improvement_path}")
    
    print("\n" + "="*80)
    print(f"ANALYSIS COMPLETED at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)