import geopandas as gpd
import pandas as pd
import os
import glob
import json
import time
from tqdm import tqdm
import multiprocessing
from functools import partial

def extract_avg_fragmentation(gpkg_file, year):
    """
    Extract average DOF (degree of fragmentation) from a GPKG file.
    
    Parameters:
    -----------
    gpkg_file : str
        Path to the GPKG file
    year : int
        Year of the data (for logging purposes)
        
    Returns:
    --------
    tuple
        (file_name, avg_dof) where avg_dof is the average dof_all value, or None if extraction fails
    """
    file_name = os.path.basename(gpkg_file)
    print(f"Processing {year} file: {file_name}")
    
    try:
        # Check if file exists
        if not os.path.exists(gpkg_file):
            print(f"  - File {file_name} does not exist.")
            return file_name, None
            
        # Read only the necessary field (dof_all) from the DOF layer
        start_time = time.time()
        dof_gdf = gpd.read_file(gpkg_file, layer='DOF', usecols=['dof_all'])
        elapsed_time = time.time() - start_time
        print(f"  - Read {len(dof_gdf)} records from {year} file (took {elapsed_time:.2f} seconds)")
        
        # Process dof_all field
        if 'dof_all' in dof_gdf.columns:
            # Convert to numeric and handle any non-numeric values
            dof_gdf['dof_all'] = pd.to_numeric(dof_gdf['dof_all'], errors='coerce').fillna(0)
            
            # Calculate average
            avg_dof = float(dof_gdf['dof_all'].mean())
            print(f"  - Average DOF for {year}: {avg_dof:.4f}")
            return file_name, avg_dof
        else:
            print(f"  - No dof_all field found in {year} file {file_name}")
            return file_name, None
            
    except Exception as e:
        print(f"Error processing {year} file {file_name}: {str(e)}")
        return file_name, None

def process_file_pair(gpkg_file_2020, base_dir_2010):
    """
    Process a pair of files (2020 and corresponding 2010) and return the results.
    
    Parameters:
    -----------
    gpkg_file_2020 : str
        Path to the 2020 GPKG file
    base_dir_2010 : str
        Directory containing 2010 GPKG files
        
    Returns:
    --------
    tuple
        (file_name, result_dict) where result_dict contains the DOF values and difference
    """
    file_name = os.path.basename(gpkg_file_2020)
    
    # Create the corresponding 2010 file path
    gpkg_file_2010 = os.path.join(base_dir_2010, file_name)
    
    # Extract average DOF for 2020
    _, avg_dof_2020 = extract_avg_fragmentation(gpkg_file_2020, 2020)
    
    # Extract average DOF for 2010 if the file exists
    avg_dof_2010 = None
    if os.path.exists(gpkg_file_2010):
        _, avg_dof_2010 = extract_avg_fragmentation(gpkg_file_2010, 2010)
    else:
        print(f"  - No corresponding 2010 file found for {file_name}")
    
    # Return results
    result = {
        "dof_all_2010": avg_dof_2010,
        "dof_all_2020": avg_dof_2020,
        "difference": None if (avg_dof_2020 is None or avg_dof_2010 is None) else (avg_dof_2020 - avg_dof_2010)
    }
    
    return file_name, result

def compare_dof_between_years(base_dir_2020, base_dir_2010, output_json, num_processes=None):
    """
    Compare average DOF values between 2010 and 2020 for all GPKG files
    using multiprocessing and output the results to a JSON file.
    
    Parameters:
    -----------
    base_dir_2020 : str
        Directory containing 2020 GPKG files
    base_dir_2010 : str
        Directory containing 2010 GPKG files
    output_json : str
        Path to output JSON file
    num_processes : int, optional
        Number of processes to use. If None, uses CPU count - 1
    """
    # Find all GPKG files in the 2020 directory
    gpkg_files_2020 = glob.glob(os.path.join(base_dir_2020, "*.gpkg"))
    print(f"Found {len(gpkg_files_2020)} GPKG files in 2020 directory")
    
    # Determine number of processes
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
    print(f"Using {num_processes} processes")
    
    # Create partial function with fixed base_dir_2010
    process_func = partial(process_file_pair, base_dir_2010=base_dir_2010)
    
    # Process files in parallel
    results = {}
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use imap_unordered to get results as they complete
        for file_name, result in tqdm(
            pool.imap_unordered(process_func, gpkg_files_2020), 
            total=len(gpkg_files_2020),
            desc="Processing GPKG files"
        ):
            results[file_name] = result
    
    # Write results to JSON file
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results written to {output_json}")
    
    # Print summary
    valid_comparisons = [v for v in results.values() if v["dof_all_2010"] is not None and v["dof_all_2020"] is not None]
    
    if valid_comparisons:
        avg_diff = sum(v["difference"] for v in valid_comparisons) / len(valid_comparisons)
        print(f"\nSummary:")
        print(f"Total files processed: {len(results)}")
        print(f"Files with both 2010 and 2020 data: {len(valid_comparisons)}")
        print(f"Average DOF change (2020 - 2010): {avg_diff:.4f}")
        
        # Count files with improvements vs. deterioration
        improved = sum(1 for v in valid_comparisons if v["difference"] < 0)
        worsened = sum(1 for v in valid_comparisons if v["difference"] > 0)
        unchanged = sum(1 for v in valid_comparisons if v["difference"] == 0)
        
        print(f"Basins with improved fragmentation (lower DOF): {improved}")
        print(f"Basins with worsened fragmentation (higher DOF): {worsened}")
        print(f"Basins with unchanged fragmentation: {unchanged}")

def main():
    # Define base directories
    base_dir_2020 = r"/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/dof_result"
    base_dir_2010 = r"/mnt/bf9340de-26bc-4032-9f11-494ba8ad1b3a/wyj/data/dof_result_2010"
    
    # Create output directory if it doesn't exist
    output_dir = '/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/wyj/大坝知识图谱材料/大坝知识图谱材料/代码/Free-Flowing-Rivers/output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output JSON file path
    output_json = os.path.join(output_dir, f"dof_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # Run the comparison with multiprocessing
    # You can specify the number of processes explicitly if needed
    # compare_dof_between_years(base_dir_2020, base_dir_2010, output_json, num_processes=4)
    compare_dof_between_years(base_dir_2020, base_dir_2010, output_json)

if __name__ == "__main__":
    # For Windows compatibility, multiprocessing needs this guard
    # This is not needed on Linux but doesn't hurt
    multiprocessing.freeze_support()
    
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")