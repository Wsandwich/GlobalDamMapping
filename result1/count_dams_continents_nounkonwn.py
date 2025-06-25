import geopandas as gpd
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import gc  # For garbage collection

# Define paths
continents_path = "E:/wyj/data/World_Continents_-530508453213467113.gpkg"
dams_dir = "E:/wyj/dam/v4/global_v5_shp"
years = [2010, 2015, 2020]

# Function to process a single year
def process_year(year, continents):
    print(f"\nProcessing dams for year {year}...")
    dam_file = os.path.join(dams_dir, f"dams_{year}.shp")
    
    # Read the dam shapefile
    dams = gpd.read_file(dam_file)
    print(f"Loaded {len(dams)} dams for {year}")
    print(f"Dam CRS: {dams.crs}")
    
    # If dams don't have CRS, try to read from PRJ file or set it from the continent file
    if dams.crs is None:
        print("Dam shapefile has no CRS defined. Checking PRJ file...")
        prj_file = os.path.join(dams_dir, f"dams_{year}.prj")
        if os.path.exists(prj_file):
            # Read the PRJ file to get the CRS
            with open(prj_file, 'r') as f:
                prj_text = f.read()
            print(f"Found PRJ file: {prj_text[:100]}...")
            try:
                # Try to set the CRS from the PRJ file
                dams = dams.set_crs(prj_text, allow_override=True)
                print(f"Set CRS from PRJ file: {dams.crs}")
            except Exception as e:
                print(f"Error setting CRS from PRJ file: {e}")
                print("Setting CRS to WGS84 (EPSG:4326) as a fallback")
                dams = dams.set_crs("EPSG:4326", allow_override=True)
        else:
            print("No PRJ file found, assuming WGS84 (EPSG:4326)")
            dams = dams.set_crs("EPSG:4326", allow_override=True)
    
    # Ensure both datasets have the same CRS
    if dams.crs != continents.crs:
        print(f"Reprojecting dams from {dams.crs} to match continent CRS ({continents.crs})")
        dams = dams.to_crs(continents.crs)
    
    # Process in chunks to avoid memory issues
    chunk_size = 10000  # Adjust based on your available memory
    total_dams = len(dams)
    num_chunks = (total_dams + chunk_size - 1) // chunk_size  # Ceiling division
    
    # Dictionary to store continent counts
    continent_counts = {}
    
    print(f"Processing {total_dams} dams in {num_chunks} chunks of size {chunk_size}...")
    
    # Create a copy of continents with simpler geometry for nearest neighbor calculations
    continent_hulls = continents.copy()
    continent_hulls['geometry'] = continents.geometry.convex_hull
    
    # Process each chunk
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_dams)
        
        print(f"Processing chunk {chunk_idx+1}/{num_chunks} (dams {start_idx} to {end_idx-1})...")
        
        # Get the current chunk of dams
        dams_chunk = dams.iloc[start_idx:end_idx].copy()
        
        # Perform spatial join for this chunk
        joined_chunk = gpd.sjoin(dams_chunk, continents, how="left", predicate="within")
        
        # Find dams with unknown continents
        unknown_dams = joined_chunk[joined_chunk["CONTINENT"].isna()].copy()
        print(f"Found {len(unknown_dams)} dams with unknown continent in this chunk")
        
        if len(unknown_dams) > 0:
            print("Classifying unknown dams based on proximity to continents...")
            
            # Process unknown dams in smaller batches to manage memory
            batch_size = 100  # Adjust as needed
            for i in range(0, len(unknown_dams), batch_size):
                batch = unknown_dams.iloc[i:i+batch_size]
                
                # Find nearest continent for each dam in this batch
                for idx, dam in tqdm(batch.iterrows(), total=len(batch), desc=f"Batch {i//batch_size+1}"):
                    # Calculate distances to all continents
                    distances = continent_hulls.geometry.distance(dam.geometry)
                    # Find the index of the minimum distance
                    min_idx = distances.idxmin()
                    # Assign the nearest continent
                    joined_chunk.loc[idx, "CONTINENT"] = continent_hulls.loc[min_idx, "CONTINENT"]
        
        # Count continents in this chunk
        chunk_counts = joined_chunk["CONTINENT"].value_counts().to_dict()
        
        # Update overall counts
        for continent, count in chunk_counts.items():
            if continent in continent_counts:
                continent_counts[continent] += count
            else:
                continent_counts[continent] = count
        
        # Save intermediate results (optional)
        output_chunk = os.path.join(os.path.dirname(dams_dir), f"dams_{year}_continents_chunk{chunk_idx}.csv")
        joined_chunk.to_csv(output_chunk)
        
        # Clean up to free memory
        del joined_chunk, dams_chunk
        if 'unknown_dams' in locals():
            del unknown_dams
        gc.collect()
    
    # Add any missing continents with zero count
    for continent in continents["CONTINENT"].unique():
        if continent not in continent_counts:
            continent_counts[continent] = 0
    
    # Count remaining unknown dams (should be 0 or minimal now)
    nan_count = continent_counts.get(np.nan, 0)
    if pd.notna(nan_count) and nan_count > 0:
        print(f"Dams that couldn't be classified: {nan_count}")
        continent_counts["Unknown"] = nan_count
        if np.nan in continent_counts:
            del continent_counts[np.nan]
    else:
        continent_counts["Unknown"] = 0
    
    # Clean up to free memory
    del dams
    gc.collect()
    
    return continent_counts

def main():
    print("Loading continent boundaries...")
    # Read continents from GeoPackage
    continents = gpd.read_file(continents_path, layer="World_Continents")
    print(f"Loaded {len(continents)} continents")
    print(f"Continent CRS: {continents.crs}")
    
    # Create a dataframe to store results
    all_continents = list(continents["CONTINENT"].unique()) + ["Unknown"]
    results = pd.DataFrame(0, index=all_continents, columns=years)
    
    # Process each year separately
    for year in years:
        # Process the year and get continent counts
        continent_counts = process_year(year, continents)
        
        # Add counts to results
        for continent, count in continent_counts.items():
            results.loc[continent, year] = count
    
    # Add total column and row
    results["Total"] = results.sum(axis=1)
    results.loc["Total"] = results.sum()
    
    # Print table
    print("\nNumber of dams in each continent by year:")
    print(results)
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(dams_dir), "dam_counts_by_continent.csv")
    results.to_csv(output_path)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()