import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import nearest_points
from shapely.strtree import STRtree  # Changed from rtree to shapely's STRtree
import time
import os
from tqdm import tqdm
import concurrent.futures
import multiprocessing
from functools import partial

# Global variables to store worker-level data
worker_rivers = None
worker_strtree = None  # Changed from worker_idx to worker_strtree

def init_worker(rivers_data):
    """
    Initialize worker process with rivers data and create a spatial index.
    This function runs once per worker process when the pool starts.
    
    Parameters:
    -----------
    rivers_data : GeoDataFrame
        The pre-loaded river network GeoDataFrame
    """
    global worker_rivers, worker_strtree
    
    # Store rivers data in worker's global namespace
    worker_rivers = rivers_data
    
    # Create STRtree spatial index for rivers in this worker process - once per worker
    start_time = time.time()
    worker_strtree = STRtree(worker_rivers.geometry.values)  # Create STRtree with geometries
    end_time = time.time()
    
    print(f"Worker process {os.getpid()} initialized with STRtree for {len(worker_rivers)} rivers in {end_time - start_time:.2f} seconds")

def process_dam_batch(batch_data, max_search_distance=None):
    """
    Process a batch of dams to find their nearest rivers.
    Uses the worker-level global variables worker_rivers and worker_strtree.
    
    Parameters:
    -----------
    batch_data : tuple
        Tuple containing (batch_start_index, dam_batch_df)
    max_search_distance : float, optional
        Maximum distance to search for nearest river
        
    Returns:
    --------
    list
        List of tuples (dam_idx, nearest_river_goid, distance_to_river)
    """
    global worker_rivers, worker_strtree
    
    batch_start, dam_batch = batch_data
    results = []
    
    for i, dam in enumerate(dam_batch.itertuples()):
        dam_idx = batch_start + i
        dam_geom = dam.geometry
        
        # Query the STRtree to find potential nearest rivers
        if max_search_distance:
            # If we have a max search distance, create a buffer around the dam
            search_area = dam_geom.buffer(max_search_distance)
            # Query STRtree for potential matches within the buffer area
            potential_matches_idx = worker_strtree.query(search_area)
        else:
            # If no max distance, query the nearest 50 river segments based on bounding box
            potential_matches_idx = worker_strtree.nearest(dam_geom, 50)
        
        if len(potential_matches_idx) == 0:
            # Skip if no potential matches found
            results.append((dam_idx, -1, -1.0))
            continue
            
        # Get the river geometries for the potential matches
        potential_rivers = worker_rivers.iloc[potential_matches_idx]
        
        # Calculate distances to potential river segments
        distances = potential_rivers.geometry.distance(dam_geom)
        
        # Find the index of the nearest river segment
        if len(distances) > 0:
            min_local_idx = distances.argmin()
            min_distance = distances.iloc[min_local_idx]
            
            # Get the GOID of the nearest river segment
            nearest_river_goid = potential_rivers.iloc[min_local_idx]['GOID']
            
            # Store the result
            results.append((dam_idx, nearest_river_goid, min_distance))
        else:
            results.append((dam_idx, -1, -1.0))
    
    return results

def find_nearest_rivers_to_dams(dam_shp_path, river_gpkg_path, output_dam_path=None, batch_size=100, max_search_distance=None, max_workers=None):
    """
    Find the nearest river segment for each dam and calculate the distance using spatial indexing and parallel processing.
    Uses a worker-level spatial index to improve efficiency.
    
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
        Maximum distance to search for nearest river. Units should match the CRS units.
    max_workers : int, optional
        Maximum number of worker processes to use. If None, uses the default value (CPU count)
        
    Returns:
    --------
    GeoDataFrame
        Updated dam GeoDataFrame with nearest river GOID and distance
    """
    start_time = time.time()
    print("Loading dam data...")
    dams = gpd.read_file(dam_shp_path)
    
    # Load rivers data once at the beginning
    print("Loading river network data (this might take a while for large datasets)...")
    river_columns = ['geometry', 'GOID', 'DIS_AV_CMS', 'RIV_ORD']
    rivers = gpd.read_file(river_gpkg_path, columns=river_columns)
    print(f"Loaded {len(rivers)} river segments")
    
    # Ensure both datasets use the same CRS
    if dams.crs != rivers.crs:
        print(f"Reprojecting dams from {dams.crs} to {rivers.crs}")
        dams = dams.to_crs(rivers.crs)
    
    # Add columns for nearest river GOID and distance if they don't exist
    if 'NEAREST_RIVER_GOID' not in dams.columns:
        dams['NEAREST_RIVER_GOID'] = -1  # Default value
    if 'DISTANCE_TO_RIVER' not in dams.columns:
        dams['DISTANCE_TO_RIVER'] = -1.0  # Default value
    
    # Prepare batches for parallel processing
    batch_data = []
    for batch_start in range(0, len(dams), batch_size):
        batch_end = min(batch_start + batch_size, len(dams))
        dam_batch = dams.iloc[batch_start:batch_end].copy()
        batch_data.append((batch_start, dam_batch))
    
    # Determine the number of workers - use all CPU cores by default
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(batch_data))
    
    print(f"Processing {len(dams)} dams in {len(batch_data)} batches with {max_workers} workers...")
    
    # Process batches in parallel using Pool with initializer
    processed_count = 0
    
    with multiprocessing.Pool(processes=max_workers, initializer=init_worker, initargs=(rivers,)) as pool:
        # Create a partial function with fixed arguments for max_search_distance
        process_batch_func = partial(
            process_dam_batch,
            max_search_distance=max_search_distance
        )
        
        # Process batches and collect results with progress tracking
        for batch_results in tqdm(pool.imap_unordered(process_batch_func, batch_data), total=len(batch_data)):
            try:
                processed_count += len(batch_results)
                
                # Update dams DataFrame with results from this batch
                for dam_idx, nearest_river_goid, distance_to_river in batch_results:
                    dams.at[dam_idx, 'NEAREST_RIVER_GOID'] = nearest_river_goid
                    dams.at[dam_idx, 'DISTANCE_TO_RIVER'] = distance_to_river
                
                # Optional: Save intermediate results periodically
                if output_dam_path and processed_count % (batch_size * 10) == 0:
                    interim_path = f"{os.path.splitext(output_dam_path)[0]}_interim.shp"
                    # save_dams_to_shapefile(dams, interim_path)
                    print(f"Saved interim results to {interim_path} after processing {processed_count} dams")
            
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
    
    elapsed_time = time.time() - start_time
    print(f"Processing complete. Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    # Save the updated dam shapefile if path is provided
    if output_dam_path:
        print(f"Saving updated dam data to {output_dam_path}")
        save_dams_to_shapefile(dams, output_dam_path)
    
    return dams

def save_dams_to_shapefile(dams, output_path):
    """
    Save the dam GeoDataFrame to a shapefile, handling non-ASCII field names.
    
    Parameters:
    -----------
    dams : GeoDataFrame
        The dam GeoDataFrame to save
    output_path : str
        Path to save the shapefile
    """
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
        mapping_path = f"{os.path.splitext(output_path)[0]}_field_mapping.csv"
        mapping_df.to_csv(mapping_path, index=False, encoding='utf-8')
        print(f"Field name mapping saved to {mapping_path}")
    
    dams_for_save.to_file(output_path)

def create_dam_river_mapping_table(dams_with_river_info, rivers, output_csv_path=None):
    """
    Create a mapping table that links dams to their nearest rivers with additional river properties.
    
    Parameters:
    -----------
    dams_with_river_info : GeoDataFrame
        Dam GeoDataFrame with NEAREST_RIVER_GOID and DISTANCE_TO_RIVER columns
    rivers : GeoDataFrame
        The pre-loaded river network GeoDataFrame
    output_csv_path : str, optional
        Path to save the mapping table as CSV
        
    Returns:
    --------
    DataFrame
        Mapping table linking dams to their nearest rivers
    """
    print("Creating mapping table...")
    
    # Get required columns from rivers
    river_columns = ['GOID', 'DIS_AV_CMS', 'RIV_ORD'] 
    river_info = rivers[river_columns].copy()
    
    # Convert dam GeoDataFrame to regular DataFrame with required columns
    dam_columns = ['NEAREST_RIVER_GOID', 'DISTANCE_TO_RIVER']
    dam_info = dams_with_river_info[dam_columns + [col for col in dams_with_river_info.columns 
                                                 if col not in dam_columns + ['geometry']]].copy()
    
    # Merge dam info with river info
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

# Example usage with parallel processing
if __name__ == "__main__":
    # Replace with your actual file paths
    dam_shp_path = "Free-Flowing-Rivers/CODE/数据/dam_shp/YELLOW_RIVER.shp"
    river_gpkg_path = "Free-Flowing-Rivers/CODE/数据/dof_result/origin_yellow.gpkg"
    output_dam_path = "Free-Flowing-Rivers/CODE/数据/dof_result/dam_with_nearest_rivers_parallel.shp"
    
    # Optional: Set a maximum search distance in the units of your CRS (e.g., meters)
    max_search_distance = 5000  # Adjust based on your data context
    
    # Process dams in parallel with worker-level spatial indices
    updated_dams = find_nearest_rivers_to_dams(
        dam_shp_path=dam_shp_path,
        river_gpkg_path=river_gpkg_path,
        output_dam_path=output_dam_path,
        batch_size=100,
        max_search_distance=max_search_distance,
        max_workers=None  # Use all available CPU cores
    )
    
    # Create a mapping table (optional)
    mapping_csv_path = "Free-Flowing-Rivers/CODE/数据/dof_result/dam_river_mapping_parallel.csv"
    
    # Load rivers if not already loaded (in case someone calls this function separately)
    river_columns = ['geometry', 'GOID', 'DIS_AV_CMS', 'RIV_ORD']
    rivers = gpd.read_file(river_gpkg_path, columns=river_columns)
    
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