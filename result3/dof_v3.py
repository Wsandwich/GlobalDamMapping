import geopandas as gpd
import os
import sys
import pickle as cPickle
import math
import numpy as np
import multiprocessing
import pandas as pd
from pathlib import Path
from functools import partial
from config import config
import tools.helper as tool


# Load global variables
fd = config.var


def calculate_DOF_vectorized(dams, streams, mode, dof_field, drf_upstream, drf_downstream, use_dam_level_df):
    """
    Calculate DOF (Degree of Fragmentation) for specified barriers with vectorized operations
    
    :param dams: GeoDataFrame containing one or more barriers
    :param streams: GeoDataFrame containing river network
    :param mode: Numeric value specifying the decay function to use
    :param dof_field: Field to fill with DOF values
    :param drf_upstream: Upstream discharge range factor
    :param drf_downstream: Downstream discharge range factor
    :param use_dam_level_df: If "YES", use DFU and DFD fields
    :return: Updated GeoDataFrame with DOF values
    """
    # Convert lists to numpy arrays for faster access
    ndoid_array = np.array(streams["NDOID"].tolist())  # Downstream reach IDs
    nuoid_list = streams["NUOID"].tolist()  # Upstream reach IDs (keep as list due to multiple values)
    wfall_array = np.array(streams["HYFALL"].tolist())  # Waterfall information
    disch_array = np.array(streams["DIS_AV_CMS"].tolist())  # Average discharge

    # Create a lookup dictionary for faster upstream reach access
    upstream_lookup = {}
    for i, nuoid_str in enumerate(nuoid_list):
        # print(nuoid_str)
        if not isinstance(nuoid_str, str):
                nuoid_str = str(nuoid_str)
        upstream_oids = [int(oid) for oid in nuoid_str.split("_") if oid and int(oid) > 0]
        upstream_lookup[i+1] = upstream_oids

    # Create numpy array for DOF values (faster updates)
    if dof_field in streams.columns:
        dof_array = streams[dof_field].values.copy()
    else:
        dof_array = np.zeros(len(streams), dtype=np.float32)
    
    # Pre-compute log10 values of discharge for efficiency
    with np.errstate(divide='ignore', invalid='ignore'):
        log10_disch = np.log10(disch_array)
        log10_disch[~np.isfinite(log10_disch)] = 0  # Handle zeros/infinities
    
    # Process each dam using batch operations where possible
    for index, dam in dams.iterrows():
        dam_goid = dam[fd.GOID]
        dam_idx = dam_goid - 1
        
        # Update discharge factors if using dam-level values
        dam_drf_upstream = drf_upstream
        dam_drf_downstream = drf_downstream
        
        if use_dam_level_df.upper() == "YES":
            dam_drf_upstream = max(dam[fd.DFU], 1.000000000000001)
            dam_drf_downstream = max(dam[fd.DFD], 1.000000000000001)
        else:
            dam_drf_upstream = max(drf_upstream, 1.000000000000001)
            dam_drf_downstream = max(drf_downstream, 1.000000000000001)
        
        # Get discharge at barrier location
        discharge_barrier = disch_array[dam_idx]
        
        # Special case: zero discharge
        if discharge_barrier == 0:
            dof_array[dam_idx] = 100
            continue
        
        # Define discharge range
        dis_low = discharge_barrier / dam_drf_upstream
        dis_high = discharge_barrier * dam_drf_downstream
        
        # Calculate upstream DOF using breadth-first search
        nodes_to_process = [dam_goid]
        visited = set()
        
        # Process upstream reaches
        while nodes_to_process:
            new_nodes = []
            for node in nodes_to_process:
                node_idx = node - 1
                
                if node in visited or wfall_array[node_idx] != 0:
                    continue
                
                visited.add(node)
                discharge_local = disch_array[node_idx]
                
                # Check if discharge is in range
                if dis_low <= discharge_local <= dis_high:
                    # Calculate DOF value
                    local_impact_score = get_dof_up(
                        discharge_local, discharge_barrier, mode, dam_drf_upstream)
                    
                    # Update DOF value if higher
                    if dof_array[node_idx] <= local_impact_score or dof_array[node_idx] == 0:
                        dof_array[node_idx] = local_impact_score
                    
                    # Add upstream reaches
                    new_nodes.extend(upstream_lookup.get(node, []))
            
            nodes_to_process = new_nodes
        
        # Calculate downstream DOF also using breadth-first search
        nodes_to_process = [dam_goid]
        visited = set()
        
        # Process downstream reaches
        while nodes_to_process:
            new_nodes = []
            for node in nodes_to_process:
                node_idx = node - 1
                
                if node in visited:
                    continue
                
                visited.add(node)
                discharge_local = disch_array[node_idx]
                
                # Check if discharge is in range
                if dis_low <= discharge_local <= dis_high:
                    # Calculate DOF value
                    local_impact_score = get_dof_down(
                        discharge_local, discharge_barrier, mode, dam_drf_downstream)
                    
                    # Update DOF value if higher
                    if dof_array[node_idx] <= local_impact_score or dof_array[node_idx] == 0:
                        dof_array[node_idx] = local_impact_score
                    
                    # Add downstream reach
                    downstream_oid = ndoid_array[node_idx]
                    if downstream_oid > 0:
                        new_nodes.append(int(downstream_oid))
            
            nodes_to_process = new_nodes
    
    # Update the GeoDataFrame with the calculated DOF values
    streams[dof_field] = dof_array
    
    return dams, streams


def get_dof_down(discharge_local, discharge_barrier, downstream_mode, dis_range_factor):
    """
    Calculate DOF for downstream reach
    :param discharge_local: Current reach discharge
    :param discharge_barrier: Barrier reach discharge
    :param downstream_mode: Downstream DOF mode
    :param dis_range_factor: Downstream discharge factor
    :return: DOF value
    """
    if discharge_local < discharge_barrier:
        discharge_local = discharge_barrier
    
    if downstream_mode == 1:
        a = abs(math.log10(discharge_local) - math.log10(discharge_barrier))
        b = a * (100 / math.log10(dis_range_factor))
        x = 100 - b
    else:
        print("Discharge mode undefined")
        sys.exit()
    
    return max(0, min(100, x))


def get_dof_up(discharge_local, discharge_barrier, upstream_mode, dis_range_factor):
    """
    Calculate DOF for upstream reach
    :param discharge_local: Current reach discharge
    :param discharge_barrier: Barrier reach discharge
    :param upstream_mode: Upstream DOF mode
    :param dis_range_factor: Upstream discharge factor
    :return: DOF value
    """
    if discharge_local > discharge_barrier:
        discharge_local = discharge_barrier
    
    if upstream_mode == 1:
        a = abs(math.log10(discharge_local) - math.log10(discharge_barrier))
        b = a * (100 / math.log10(dis_range_factor))
        x = 100 - b
    else:
        print("Discharge mode undefined")
        sys.exit()
    
    return max(0, min(100, x))


def load_streams(stream_table, dof_field):
    """
    Load river data and add DOF field
    :param stream_table: GeoDataFrame representing rivers
    :param dof_field: Field to store DOF results
    :return: Updated GeoDataFrame
    """
    flds = [fd.BAS_ID, fd.GOID, fd.NOID, fd.NDOID, fd.NUOID, fd.RIV_ORD, fd.DIS_AV_CMS, fd.HYFALL]
    tool.check_fields(stream_table, flds)
    
    # Fill NA values
    for field in flds:
        if field in stream_table.columns:
            stream_table[field] = stream_table[field].fillna(0)
    
    # Add DOF field if it doesn't exist
    if dof_field not in stream_table.columns:
        stream_table[dof_field] = 0.0
    
    return stream_table


def load_dams(dam_table, use_dam_level_df):
    """
    Load dam data from database
    :param dam_table: Dam data table
    :param use_dam_level_df: Whether to use DFU and DFD fields
    :return: GeoDataFrame
    """
    # Map NEAREST_RI to GOID directly during dam loading
    if 'NEAREST_RI' in dam_table.columns and fd.GOID not in dam_table.columns:
        print("Mapping NEAREST_RI to GOID for dam table")
        dam_table[fd.GOID] = dam_table['NEAREST_RI']
    
    # Define required fields
    if use_dam_level_df.lower() == "yes":
        flds = [fd.BAS_ID, fd.GOID, fd.STOR_MCM, fd.DFU, fd.DFD]
    else:
        flds = [fd.GOID]
    
    # Check if fields exist
    available_fields = [f for f in flds if f in dam_table.columns]
    tool.check_fields(dam_table, available_fields)
    
    # Fill NA values for available fields
    for field in available_fields:
        dam_table[field] = dam_table[field].fillna(0)
    
    # Ensure GOID is an integer
    if fd.GOID in dam_table.columns:
        dam_table[fd.GOID] = dam_table[fd.GOID].astype(int)
    
    return dam_table


def run_basin(streams, dams, basin, stamp, scratchws, dof_field, drf_upstream, drf_downstream, mode, use_dam_level_df):
    """
    Calculate DOF for a specific basin
    :param streams: River data
    :param dams: Dam data
    :param basin: Basin name/ID
    :param stamp: Timestamp
    :param scratchws: Temporary workspace
    :param dof_field: Field to fill DOF values
    :param drf_upstream: Upstream discharge factor
    :param drf_downstream: Downstream discharge factor
    :param mode: DOF mode
    :param use_dam_level_df: Whether to use barrier level discharge factors
    :return: Result path
    """
    # Set environment
    temp_out_folder = set_environment(scratchws, basin, stamp)
    
    # Update routing indices
    streams = tool.update_stream_routing_index(streams)
    dams = tool.update_dam_routing_index(dams, streams)
    
    # Calculate DOF using vectorized function
    dams, streams = calculate_DOF_vectorized(
        dams, streams, mode, dof_field, drf_upstream, drf_downstream, use_dam_level_df)
    
    # Export results
    final_table = export(streams, basin, temp_out_folder)
    return final_table


def set_environment(scratch_ws, basin, stamp):
    """
    Create separate output path for specific basin
    :param scratch_ws: Temporary workspace
    :param basin: Basin
    :param stamp: Timestamp
    :return: Full output path
    """
    out_folder = f"basin_{basin}_{stamp}"
    fullpath = os.path.join(scratch_ws, out_folder)
    os.makedirs(fullpath, exist_ok=True)
    return fullpath


def export(streams, basin, folder):
    """
    Export DOF calculation results
    :param streams: Calculation results
    :param basin: Basin
    :param folder: Target folder
    :return: Result file path
    """
    suffix = ".bas"
    name = f"out_{basin}"
    fullname = f"{name}{suffix}"
    fullpath = os.path.join(folder, fullname)
    
    # Save as pickle file
    tool.save_as_cpickle(pickle_object=streams, folder=folder, name=name, file_extension=suffix)
    return fullpath


def run_dof(stamp=None):
    """
    Main function for multiprocessing DOF calculation for all basins
    :param stamp: Timestamp
    :return: Results
    """
    if stamp is None:
        stamp = tool.get_stamp()

    dam_folder = '/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/dams_with_rivers_03'
    streams_folder = '/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/rivers_by_basin_03'
    output_path = r"/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/dof_result"

    dam_files = [f for f in os.listdir(dam_folder) if f.endswith('.shp')]

    # Process all stream files
    for streams_file in os.listdir(streams_folder):
        if not streams_file.endswith('.gpkg'):
            continue

        basin_name = streams_file.replace('.gpkg', '')
        dam_file = f"{basin_name}.shp"
        
        streams_path = os.path.join(streams_folder, streams_file)
        new_gpkg_name = streams_file
        gpkg_full_path = os.path.join(output_path, new_gpkg_name)


    
        if os.path.exists(gpkg_full_path):
            print(f"Output file {gpkg_full_path} already exists - skipping...")
            continue

        # Check if matching dam file exists
        dam_path = os.path.join(dam_folder, dam_file)
        if not os.path.exists(dam_path):
            print(f"Dam file {dam_path} does not exist - creating output with DOF=0...")
            
            # Load streams data
            streams_fc = gpd.read_file(streams_path)
            dof_field = 'dof_all'
            
            # Add DOF field with all zeros
            if dof_field not in streams_fc.columns:
                streams_fc[dof_field] = 0.0
                
            # Save to output file
            os.makedirs(os.path.dirname(gpkg_full_path), exist_ok=True)
            streams_fc.to_file(gpkg_full_path, driver="GPKG", layer="DOF")
            
            # Also create COMBINED_RESULT layer
            streams_fc.to_file(gpkg_full_path, driver="GPKG", layer="COMBINED_RESULT")
            continue

        # Parallel processing setup
        num_cores = 35  # Limit to prevent excessive memory usage
        print(f"Number of CPU cores to use: {num_cores}")

        dof_field = 'dof_all'
        dof_mode = 1
        drf_upstream = 5
        drf_downstream = 5
        use_dam_level_df = "No"
        update_stream_mode = "Yes"
        
        # Create output folders
        output_folder = os.path.join(output_path, f"Results_{stamp}")
        scratch_ws = os.path.join(output_folder, "Scratch")
        
        # Ensure paths exist
        Path(scratch_ws).mkdir(parents=True, exist_ok=True)
        
        print(f"Discharge range factor used (upstream): {drf_upstream}")
        print(f"Discharge range factor used (downstream): {drf_downstream}")
        print("Loading streams and dams...")
        
        # Load data
        dam_fc = gpd.read_file(dam_path)
        streams_fc = gpd.read_file(streams_path)
        
        # Print dam data structure for debugging
        print("Dam data columns:", dam_fc.columns.tolist())
        if 'NEAREST_RI' in dam_fc.columns:
            print(f"Found NEAREST_RI column - will map to {fd.GOID}")
        
        streams = load_streams(streams_fc.copy(), dof_field)
        dams_temp = load_dams(dam_fc, use_dam_level_df)
        
        # Determine basin
        basin = 'YELLOW'  # Change to match actual basin ID
        

        
        # Execute calculation
        if __name__ == '__main__' and num_cores > 1:
            # Use multiprocessing
            print("Starting analysis with multiprocessing...")
            
            with multiprocessing.Pool(num_cores) as pool:
                run_basin_partial = partial(
                    run_basin,
                    streams.copy(), dams_temp.copy(), basin,
                    stamp, scratch_ws, dof_field,
                    drf_upstream, drf_downstream,
                    dof_mode, use_dam_level_df
                )
                
                # Create multiple jobs by splitting the dams
                # This approach creates multiple smaller tasks for better load balancing
                chunk_size = max(1, len(dams_temp) // num_cores)
                dam_chunks = [dams_temp.iloc[i:i + chunk_size] for i in range(0, len(dams_temp), chunk_size)]
                
                jobs = []
                for i, dam_chunk in enumerate(dam_chunks):
                    jobs.append(pool.apply_async(
                        run_basin,
                        (
                            streams.copy(), dam_chunk, f"{basin}_{i}",
                            stamp, scratch_ws, dof_field,
                            drf_upstream, drf_downstream,
                            dof_mode, use_dam_level_df
                        )
                    ))
                
                pool.close()
                pool.join()
                out_basin = [job.get() for job in jobs]
        else:
            # Run in single process mode
            print("Starting analysis in single process mode...")
            result = run_basin(
                streams, dams_temp, basin,
                stamp, scratch_ws, dof_field,
                drf_upstream, drf_downstream,
                dof_mode, use_dam_level_df
            )
            out_basin = [result]

        print(f"Merging temporary outputs into output table {gpkg_full_path} ...")
        
        # Load and merge results
        tbl = {}
        for i, bas in enumerate(out_basin):
            with open(bas, 'rb') as fp:
                tbl[i] = cPickle.load(fp)
        
        # Concatenate results
        if len(tbl) > 1:
            # Combine results from multiple processes
            # Create a single result DataFrame
            gdf = pd.concat([df[dof_field] for df in tbl.values()], axis=1)
            
            # Take maximum DOF value for each row
            result_dof = gdf.max(axis=1)
            
            # Create final GeoDataFrame
            final_gdf = streams.copy()
            final_gdf[dof_field] = result_dof
        else:
            # Single process result
            final_gdf = next(iter(tbl.values()))
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(gpkg_full_path), exist_ok=True)
        
        # Save results to new file
        print(f"Saving results to new file: {gpkg_full_path}")
        final_gdf.to_file(gpkg_full_path, driver="GPKG", layer="DOF")
        
        # Add index for faster queries
        tool.add_index(lyr=final_gdf, field_name="GOID")
        
        # Update stream data if requested
        if update_stream_mode.lower() == "yes":
            print(f"Combining original data with new {dof_field} values")
            
            # Create a copy of the original data
            result_gdf = streams_fc.copy()
            
            # Create mapping relationship and merge fields
            dof_values = final_gdf[[dof_field]].copy()
            
            # Merge on GOID
            result_gdf = result_gdf.merge(dof_values, left_on="GOID", right_index=True, how="left")
            
            # Save as new layer
            result_gdf.to_file(gpkg_full_path, driver="GPKG", layer="COMBINED_RESULT")
        
        # Clean up temporary files
        try:
            import shutil
            shutil.rmtree(scratch_ws)
        except Exception as e:
            print(f"Warning: Could not clean up scratch workspace: {e}")
        
        # return final_gdf[dof_field]


if __name__ == "__main__":
    stamp = tool.get_stamp()
    run_dof(stamp)