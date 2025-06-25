import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.strtree import STRtree
import time
import os
from tqdm import tqdm
import multiprocessing
from functools import partial
from pathlib import Path

def find_nearest_rivers_to_dams_single_process(dam_shp_path, river_gpkg_path, output_dam_path=None, 
                                               batch_size=100, max_search_distance=None):
    """
    A simplified version that doesn't use internal multiprocessing
    """
    start_time = time.time()
    print(f"Processing {dam_shp_path}...")
    dams = gpd.read_file(dam_shp_path)
    
    print(f"Loading river network from {river_gpkg_path}...")
    river_columns = ['geometry', 'GOID', 'DIS_AV_CMS', 'RIV_ORD']
    rivers = gpd.read_file(river_gpkg_path, columns=river_columns)
    print(f"Loaded {len(rivers)} river segments")
    
    # Ensure CRS compatibility
    if dams.crs is None:
        print(f"Warning: Dam shapefile has no CRS defined. Assuming same CRS as rivers.")
        dams.set_crs(rivers.crs, inplace=True)
    elif dams.crs != rivers.crs:
        print(f"Reprojecting dams from {dams.crs} to {rivers.crs}")
        dams = dams.to_crs(rivers.crs)
    
    # Add columns for nearest river GOID and distance
    if 'NEAREST_RIVER_GOID' not in dams.columns:
        dams['NEAREST_RIVER_GOID'] = -1
    if 'DISTANCE_TO_RIVER' not in dams.columns:
        dams['DISTANCE_TO_RIVER'] = -1.0
    
    # Create spatial index for rivers - once for all dams
    print(f"Building spatial index for {len(rivers)} river segments...")
    river_strtree = STRtree(rivers.geometry.values)
    
    # Process each dam (no internal multiprocessing)
    print(f"Processing {len(dams)} dams...")
    for i, dam in tqdm(enumerate(dams.itertuples()), total=len(dams)):
        dam_geom = dam.geometry
        
        # Query the STRtree to find potential nearest rivers
        if max_search_distance:
            search_area = dam_geom.buffer(max_search_distance)
            potential_matches_idx = river_strtree.query(search_area)
        else:
            potential_matches_idx = river_strtree.nearest(dam_geom, 50)
        
        if len(potential_matches_idx) == 0:
            continue
            
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
            
            # Update the dam with the nearest river info
            dams.at[i, 'NEAREST_RIVER_GOID'] = nearest_river_goid
            dams.at[i, 'DISTANCE_TO_RIVER'] = min_distance
    
    # Save the updated dam shapefile if path is provided
    if output_dam_path:
        print(f"Saving updated dam data to {output_dam_path}")
        dams.to_file(output_dam_path)
        print(f"Successfully saved file to {output_dam_path}")
    
    elapsed_time = time.time() - start_time
    print(f"Processing complete. Total time: {elapsed_time:.2f} seconds")
    
    return dams

def process_basin(basin_id, rivers_dir, dams_dir, output_dams_dir, output_mapping_dir, max_search_distance):
    """
    Process a single basin
    """
    try:
        # Construct file paths
        river_gpkg_path = os.path.join(rivers_dir, f"{basin_id}.gpkg")
        dam_shp_path = os.path.join(dams_dir, f"{basin_id}.shp")
        
        # Check if files exist
        if not os.path.exists(river_gpkg_path):
            print(f"Basin {basin_id}: River file does not exist")
            return {"basin_id": basin_id, "status": "skipped", "reason": "River file does not exist"}
        if not os.path.exists(dam_shp_path):
            print(f"Basin {basin_id}: Dam file does not exist")
            return {"basin_id": basin_id, "status": "skipped", "reason": "Dam file does not exist"}
        
        # Construct output paths
        output_dam_path = os.path.join(output_dams_dir, f"{basin_id}.shp")
        mapping_csv_path = os.path.join(output_mapping_dir, f"{basin_id}.csv")
        
        # Skip if already processed
        if os.path.exists(output_dam_path):
            print(f"Basin {basin_id}: Output file already exists, skipping")
            return {"basin_id": basin_id, "status": "skipped", "reason": "Output file already exists"}
        
        # Process dams in this basin (using the single-process version)
        updated_dams = find_nearest_rivers_to_dams_single_process(
            dam_shp_path=dam_shp_path,
            river_gpkg_path=river_gpkg_path,
            output_dam_path=output_dam_path,
            batch_size=100,
            max_search_distance=max_search_distance
        )
        
        # Create mapping table
        river_columns = ['geometry', 'GOID', 'DIS_AV_CMS', 'RIV_ORD']
        rivers = gpd.read_file(river_gpkg_path, columns=river_columns)
        
        # Generate mapping table
        dam_columns = ['NEAREST_RIVER_GOID', 'DISTANCE_TO_RIVER']
        dam_info = updated_dams[dam_columns + [col for col in updated_dams.columns 
                                            if col not in dam_columns + ['geometry']]].copy()
        
        river_info = rivers[['GOID', 'DIS_AV_CMS', 'RIV_ORD']].copy()
        
        mapping_table = pd.merge(
            dam_info,
            river_info,
            left_on='NEAREST_RIVER_GOID',
            right_on='GOID',
            how='left'
        )
        
        # Save mapping table
        print(f"Saving mapping table to {mapping_csv_path}")
        mapping_table.to_csv(mapping_csv_path, index=False)
        print(f"Successfully saved mapping table to {mapping_csv_path}")
        
        # Collect statistics
        total_dams = len(updated_dams)
        valid_distances = updated_dams['DISTANCE_TO_RIVER'][updated_dams['DISTANCE_TO_RIVER'] >= 0]
        valid_matches = len(valid_distances)
        distances_list = valid_distances.tolist()
        
        print(f"Basin {basin_id} processing complete: {valid_matches}/{total_dams} dams matched to rivers")
        
        return {
            "basin_id": basin_id,
            "status": "success",
            "total_dams": total_dams,
            "valid_matches": valid_matches,
            "distances": distances_list
        }
    
    except Exception as e:
        import traceback
        print(f"Basin {basin_id} processing failed with error: {str(e)}")
        print(traceback.format_exc())
        return {"basin_id": basin_id, "status": "failed", "reason": str(e)}

def main():
    # Input and output directories
    rivers_dir = "/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/rivers_by_basin_03"
    dams_dir = "/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/Controlled_experiment/dam/GOODD"
    
    output_dams_dir = "/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/Controlled_experiment/dam_with_rivers/GOODD_with_rivers_03"
    output_mapping_dir = "/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/Controlled_experiment/dam_with_rivers/GOODD_river_mappings_03"
    
    # Create output directories if they don't exist
    os.makedirs(output_dams_dir, exist_ok=True)
    os.makedirs(output_mapping_dir, exist_ok=True)
    
    # Maximum search distance
    max_search_distance = 5000
    
    # Get all basin IDs
    river_files = [f for f in os.listdir(rivers_dir) if f.endswith('.gpkg')]
    basin_ids = [Path(f).stem for f in river_files]
    print(f"Found {len(basin_ids)} basins to process")
    
    # Use multiprocessing to process basins in parallel (without nested multiprocessing)
    # num_processes = min(os.cpu_count(), 4)  # Limit to prevent memory issues
    num_processes = 20
    print(f"Using {num_processes} parallel processes")
    
    # Create partial function with fixed arguments
    process_basin_partial = partial(
        process_basin,
        rivers_dir=rivers_dir,
        dams_dir=dams_dir,
        output_dams_dir=output_dams_dir,
        output_mapping_dir=output_mapping_dir,
        max_search_distance=max_search_distance
    )
    
    # Process basins in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_basin_partial, basin_ids),
            total=len(basin_ids),
            desc="Processing basins"
        ))
    
    # Summarize results
    success_count = 0
    skip_count = 0
    fail_count = 0
    total_dams = 0
    total_matches = 0
    all_distances = []
    
    for result in results:
        if result["status"] == "success":
            success_count += 1
            total_dams += result["total_dams"]
            total_matches += result["valid_matches"]
            all_distances.extend(result["distances"])
        elif result["status"] == "skipped":
            skip_count += 1
        elif result["status"] == "failed":
            fail_count += 1
    
    # Print final statistics
    print("\nProcessing Summary:")
    print(f"Basins processed successfully: {success_count}")
    print(f"Basins skipped: {skip_count}")
    print(f"Basins failed: {fail_count}")
    print(f"Total dams processed: {total_dams}")
    print(f"Total successful river matches: {total_matches}")
    
    if all_distances:
        print(f"Average distance: {np.mean(all_distances):.2f} units")
        print(f"Maximum distance: {np.max(all_distances):.2f} units")
        print(f"Minimum distance: {np.min(all_distances):.2f} units")

if __name__ == "__main__":
    main()